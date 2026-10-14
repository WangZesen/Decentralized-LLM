"""Bounded positional network reads with ordered delivery and asynchronous transfer."""

import os
import queue
import resource
import threading
import time
from bisect import bisect_right
from collections import OrderedDict, deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np
import torch

from decentralized_llm.cache import TokenCache
from decentralized_llm.config import Config
from decentralized_llm.sampling import PRELOAD_BLOCKS, BatchSpec, GlobalSampler, check_memory
from decentralized_llm.state import Batch, IndexArray, TokenArray


@dataclass
class ReadTiming:
    shard: str
    byte_offset: int
    size: int
    sample: int
    open_ms: float
    read_ms: float
    total_ms: float


@dataclass
class LoadedBatch:
    spec: BatchSpec
    tokens: TokenArray
    created_at: float = field(default_factory=time.perf_counter)
    ready_at: float = 0.0
    slowest_read: ReadTiming | None = None


@dataclass
class PendingBatch:
    batch: LoadedBatch
    remaining: int


@dataclass
class Segment:
    path: Path
    offset: int
    size: int
    destination: TokenArray
    destination_offset: int
    sample: int
    batch: LoadedBatch | None = None


@dataclass
class ResidentRead:
    segments: list[Segment]
    rows: IndexArray


class ReadCancelled(Exception):
    pass


class PositionalReader:
    def __init__(self, config: Config, cancel: threading.Event):
        self.config = config.data
        self.cancel = cancel
        self.local = threading.local()
        self.lock = threading.Lock()
        self.handles: list[OrderedDict[Path, int]] = []
        self.requested_bytes = 0
        self.read_bytes = 0
        self.read_calls = 0
        self.slowest_read: ReadTiming | None = None
        soft_limit = resource.getrlimit(resource.RLIMIT_NOFILE)[0]
        self.descriptor_limit = min(64, max(1, (soft_limit - 64) // config.data.reader_threads))

    def descriptor(self, path: Path) -> int:
        if not hasattr(self.local, "handles"):
            self.local.handles = OrderedDict()
            with self.lock:
                self.handles.append(self.local.handles)
        handles: OrderedDict[Path, int] = self.local.handles
        if path in handles:
            handles.move_to_end(path)
            return handles[path]
        if len(handles) == self.descriptor_limit:
            _, descriptor = handles.popitem(last=False)
            os.close(descriptor)
        descriptor = os.open(path, os.O_RDONLY)
        handles[path] = descriptor
        return descriptor

    def read_group(self, segments: list[Segment]) -> None:
        if self.cancel.is_set():
            raise ReadCancelled()
        path = segments[0].path
        offset = segments[0].offset
        stop = max(s.offset + s.size for s in segments)
        started = time.perf_counter()
        try:
            descriptor = self.descriptor(path)
            opened = time.perf_counter()
            data = bytearray(stop - offset)
            filled = 0
            while filled < len(data):
                if self.cancel.is_set():
                    raise ReadCancelled()
                chunk = os.pread(descriptor, len(data) - filled, offset + filled)
                if not chunk:
                    raise OSError(f"short read/EOF at byte {offset + filled}")
                data[filled : filled + len(chunk)] = chunk
                filled += len(chunk)
                with self.lock:
                    self.read_bytes += len(chunk)
                    self.read_calls += 1
            read_finished = time.perf_counter()
            for segment in segments:
                relative = segment.offset - offset
                values = np.frombuffer(data, dtype="<u2", count=segment.size // 2, offset=relative)
                start = segment.destination_offset
                segment.destination[start : start + len(values)] = values
            with self.lock:
                self.requested_bytes += sum(s.size for s in segments)
                elapsed_ms = (time.perf_counter() - started) * 1000
                batch = segments[0].batch
                reader_peak = self.slowest_read is None or elapsed_ms > self.slowest_read.total_ms
                batch_peak = batch is not None and (
                    batch.slowest_read is None or elapsed_ms > batch.slowest_read.total_ms
                )
                if reader_peak or batch_peak:
                    timing = ReadTiming(
                        str(path),
                        offset,
                        stop - offset,
                        segments[0].sample,
                        (opened - started) * 1000,
                        (read_finished - opened) * 1000,
                        elapsed_ms,
                    )
                    if reader_peak:
                        self.slowest_read = timing
                    if batch_peak and batch is not None:
                        batch.slowest_read = timing
        except ReadCancelled:
            raise
        except Exception as error:
            samples = [s.sample for s in segments[:8]]
            raise OSError(f"read failed: shard={path}, samples={samples}, byte_offset={offset}: {error}") from error

    def groups(self, segments: list[Segment], *, amplification: float | None = None) -> list[list[Segment]]:
        amplification = self.config.max_read_amplification if amplification is None else amplification
        segments.sort(key=lambda s: (s.path, s.offset))
        groups: list[list[Segment]] = []
        requested = 0
        end = 0
        for segment in segments:
            size = max(end, segment.offset + segment.size) - (groups[-1][0].offset if groups else 0)
            if (
                groups
                and segment.path == groups[-1][0].path
                and segment.offset - end <= self.config.coalesce_gap_bytes
                and size <= self.config.read_range_bytes
                and size <= amplification * (requested + segment.size)
            ):
                groups[-1].append(segment)
                requested += segment.size
                end = max(end, segment.offset + segment.size)
            else:
                groups.append([segment])
                requested = segment.size
                end = segment.offset + segment.size
        return groups

    def close(self) -> None:
        # Called only after all running reads have returned.
        for handles in self.handles:
            for descriptor in handles.values():
                os.close(descriptor)
            handles.clear()


class TokenLoader:
    def __init__(self, cache: TokenCache, config: Config, rank: int, cursor: int, stop: int):
        if not 0 <= rank < config.distributed.world_size or cursor < 0 or cursor > stop:
            raise ValueError("invalid rank or sampling cursor")
        self.memory = check_memory(cache, config, stop=stop)
        self.mode = "resident" if self.memory["resident_tokens"] else "streaming"
        self.cache, self.config = cache, config
        self.sampler = GlobalSampler.from_config(cache, config)
        self.committed = cursor
        self.delivered = cursor
        self.specs = self.sampler.batches(config, rank, cursor, stop)
        self.cancel = threading.Event()
        self.ready = threading.Event()
        self.changed = threading.Event()
        self.started = time.perf_counter()
        self.startup_seconds = 0.0
        self.preload_seconds = 0.0
        self.failure: BaseException | None = None
        self.startup_read_bytes = 0
        self.startup_read_calls = 0
        self.resident_ready = threading.Event()
        self.resident_loaded_blocks = 0
        self.resident_remaining = np.empty(0, dtype=np.int64)
        self.resident_indices = np.empty(0, dtype=np.int64)
        self.resident_tokens = np.empty((0, config.model.context_length + 1), dtype="<u2")
        self.published = 0
        self.slots = threading.BoundedSemaphore(config.data.prefetch_batches)
        self.items: queue.Queue[LoadedBatch | BaseException | None] = queue.Queue(config.data.prefetch_batches)
        self.reader = PositionalReader(config, self.cancel)
        self.pool = ThreadPoolExecutor(max_workers=config.data.reader_threads, thread_name_prefix="token-read")
        self.waits: deque[float] = deque(maxlen=8192)
        self.occupancies: deque[int] = deque(maxlen=8192)
        self.wait_total = 0.0
        self.largest_waits: list[dict] = []
        self.starved = 0
        self.batches = 0
        self.thread = threading.Thread(target=self._produce, name="token-prefetch", daemon=True)
        self.thread.start()

    def _put(self, item: LoadedBatch | BaseException | None) -> bool:
        while not self.cancel.is_set():
            try:
                self.items.put(item, timeout=0.1)
                return True
            except queue.Full:
                pass
        return False

    def _mark_ready(self) -> None:
        if not self.ready.is_set():
            with self.reader.lock:
                self.startup_read_bytes = self.reader.read_bytes
                self.startup_read_calls = self.reader.read_calls
            self.startup_seconds = time.perf_counter() - self.started
            self.ready.set()

    def _publish(self, batch: LoadedBatch) -> bool:
        batch.ready_at = time.perf_counter()
        if not self._put(batch):
            return False
        self.published += 1
        if self.published >= self.config.data.prefetch_batches:
            self._mark_ready()
        return True

    def _read_completed(self, _: Future[None]) -> None:
        self.changed.set()

    def _sample_segments(self, sample: int, destination: TokenArray, batch: LoadedBatch | None = None) -> list[Segment]:
        segments: list[Segment] = []
        length = self.config.model.context_length
        offsets, paths = self.cache.offsets["train"], self.cache.paths["train"]
        max_tokens = self.config.data.read_range_bytes // 2
        start = sample * length
        end = start + length + 1
        dest = 0
        while start < end:
            shard = bisect_right(offsets, start) - 1
            stop = min(end, offsets[shard + 1], start + max_tokens)
            segments.append(
                Segment(
                    paths[shard], (start - offsets[shard]) * 2, (stop - start) * 2, destination, dest, sample, batch
                )
            )
            dest += stop - start
            start = stop
        return segments

    def _segments(self, batches: list[LoadedBatch]) -> list[Segment]:
        return [
            segment
            for batch in batches
            for row, sample in enumerate(batch.spec.indices)
            for segment in self._sample_segments(int(sample), batch.tokens[row], batch)
        ]

    def _resident_groups(self, rows: IndexArray, batch: LoadedBatch | None = None) -> list[ResidentRead]:
        # Only the producer mutates row state. A row is claimed exactly once;
        # split reads must all finish before anyone can gather it.
        segments = []
        for row in np.unique(rows):
            if self.resident_remaining[row] != -1:
                continue
            parts = self._sample_segments(int(self.resident_indices[row]), self.resident_tokens[row], batch)
            self.resident_remaining[row] = len(parts)
            segments.extend(parts)
        amplification = None if batch is not None else self.config.data.preload_max_read_amplification
        return [
            ResidentRead(group, np.searchsorted(self.resident_indices, [s.sample for s in group]))
            for group in self.reader.groups(segments, amplification=amplification)
        ]

    def _produce_resident(self) -> None:
        """Prioritize upcoming batches while filling the remaining cache in the background."""
        self.sampler.block(0)  # Initialize the fixed pool without changing its identity.
        count = self.memory["resident_indices"] // 8
        if self.sampler.pool.size:
            self.resident_indices = np.sort(self.sampler.pool)
        elif count < self.sampler.blocks:
            self.resident_indices = np.sort(self.sampler.permutation[:count])
        else:
            self.resident_indices = np.arange(count, dtype=np.int64)
        self.resident_indices.flags.writeable = False
        self.resident_tokens = np.empty((count, self.config.model.context_length + 1), dtype="<u2")
        self.resident_remaining = np.full(count, -1, dtype=np.int64)
        pending: dict[Future[None], ResidentRead] = {}
        foreground: deque[ResidentRead] = deque()
        background: deque[ResidentRead] = deque()
        batches: deque[tuple[LoadedBatch, IndexArray]] = deque()
        background_cursor = 0
        exhausted = False
        workers = self.config.data.reader_threads
        try:
            while not self.cancel.is_set():
                self.changed.clear()
                for future in tuple(pending):
                    if future.done():
                        future.result()
                        work = pending.pop(future)
                        np.subtract.at(self.resident_remaining, work.rows, 1)
                        self.resident_loaded_blocks += int(
                            np.count_nonzero(self.resident_remaining[np.unique(work.rows)] == 0)
                        )
                if self.resident_loaded_blocks == count and not self.resident_ready.is_set():
                    self.pool.shutdown(wait=True)
                    self.reader.close()
                    self.resident_tokens.flags.writeable = False
                    self.preload_seconds = time.perf_counter() - self.started
                    self.resident_ready.set()
                while not exhausted and self.slots.acquire(blocking=False):
                    spec = next(self.specs, None)
                    if spec is None:
                        self.slots.release()
                        exhausted = True
                        break
                    batch = LoadedBatch(
                        spec, np.empty((len(spec.indices), self.config.model.context_length + 1), dtype="<u2")
                    )
                    rows = np.searchsorted(self.resident_indices, spec.indices)
                    batches.append((batch, rows))
                    # Promote queued background groups needed by this batch.
                    # Running reads already own their rows; never submit duplicates.
                    needed = set(rows.tolist())
                    for _ in range(len(background)):
                        work = background.popleft()
                        if any(row in needed for row in work.rows):
                            foreground.append(work)
                        else:
                            background.append(work)
                    foreground.extend(self._resident_groups(rows, batch))
                while batches and np.all(self.resident_remaining[batches[0][1]] == 0):
                    batch, rows = batches.popleft()
                    np.copyto(batch.tokens, self.resident_tokens[rows])
                    if not self._publish(batch):
                        return
                if exhausted and not batches:
                    self._mark_ready()
                    self._put(None)
                    return
                # Initial queue fill uses every reader. Afterwards leave one worker
                # available for demand reads; background work never builds an
                # unbounded executor queue. One-worker configurations serialize I/O.
                while foreground and len(pending) < workers:
                    work = foreground.popleft()
                    future = self.pool.submit(self.reader.read_group, work.segments)
                    pending[future] = work
                    future.add_done_callback(self._read_completed)
                if self.ready.is_set() and not foreground:
                    if not background and not pending:
                        while background_cursor < count and not background:
                            stop = min(background_cursor + PRELOAD_BLOCKS, count)
                            rows = np.arange(background_cursor, stop, dtype=np.int64)
                            background.extend(self._resident_groups(rows))
                            background_cursor = stop
                    while background and len(pending) < max(1, workers - 1):
                        work = background.popleft()
                        future = self.pool.submit(self.reader.read_group, work.segments)
                        pending[future] = work
                        future.add_done_callback(self._read_completed)
                self.changed.wait()
        finally:
            for future in pending:
                future.cancel()

    def _produce_streaming(self) -> None:
        pending: dict[Future[None], PendingBatch] = {}
        batches: deque[PendingBatch] = deque()
        exhausted = False
        try:
            while not self.cancel.is_set():
                # Clear before examining state, so completions/consumption during
                # this iteration cannot lose their wakeup.
                self.changed.clear()
                while not exhausted and self.slots.acquire(blocking=False):
                    spec = next(self.specs, None)
                    if spec is None:
                        self.slots.release()
                        exhausted = True
                        break
                    batch = LoadedBatch(
                        spec, np.empty((len(spec.indices), self.config.model.context_length + 1), dtype="<u2")
                    )
                    groups = self.reader.groups(self._segments([batch]))
                    entry = PendingBatch(batch, len(groups))
                    batches.append(entry)
                    # FIFO submission prioritizes earlier batches. Coalescing is
                    # confined to a batch so a later read cannot hold it hostage.
                    for group in groups:
                        future = self.pool.submit(self.reader.read_group, group)
                        pending[future] = entry
                        future.add_done_callback(self._read_completed)
                for future in tuple(pending):
                    if future.done():
                        future.result()
                        pending.pop(future).remaining -= 1
                while batches and batches[0].remaining == 0:
                    if not self._publish(batches.popleft().batch):
                        return
                if exhausted and not batches:
                    self._mark_ready()
                    self._put(None)
                    return
                self.changed.wait()
        finally:
            for future in pending:
                future.cancel()

    def _produce(self) -> None:
        try:
            if self.mode == "resident":
                self._produce_resident()
            else:
                self._produce_streaming()
        except ReadCancelled:
            pass
        except BaseException as error:
            self.failure = error
            self._put(error)
        finally:
            self._mark_ready()

    def warmup(self, *, resident: bool = False) -> None:
        self.ready.wait()
        if resident and self.mode == "resident":
            while not self.resident_ready.wait(timeout=0.1):
                if self.failure is not None:
                    raise self.failure
                if self.cancel.is_set():
                    raise ReadCancelled("resident warmup cancelled")

    def next(self) -> LoadedBatch:
        start = time.perf_counter()
        occupancy = self.items.qsize()
        item = self.items.get()
        elapsed = time.perf_counter() - start
        if isinstance(item, BaseException):
            raise item
        if item is None:
            raise StopIteration
        self.slots.release()
        self.changed.set()
        self.occupancies.append(occupancy)
        self.waits.append(elapsed)
        self.wait_total += elapsed
        if len(self.largest_waits) < 8 or elapsed * 1000 > self.largest_waits[-1]["wait_ms"]:
            self.largest_waits.append(
                {
                    "wait_ms": elapsed * 1000,
                    "global_start": item.spec.start,
                    "global_stop": item.spec.stop,
                    "queue_occupancy": occupancy,
                    "batch_prepare_ms": (item.ready_at - item.created_at) * 1000,
                    "producer_ready_after_request_ms": max(0.0, item.ready_at - start) * 1000,
                    "after_ready_ms": max(0.0, start + elapsed - max(item.ready_at, start)) * 1000,
                    "slowest_read": asdict(item.slowest_read) if item.slowest_read is not None else None,
                }
            )
            self.largest_waits.sort(key=lambda event: event["wait_ms"], reverse=True)
            del self.largest_waits[8:]
        self.batches += 1
        self.starved += int(occupancy == 0)
        self.delivered = item.spec.stop
        return item

    def commit(self, cursor: int) -> None:
        if not self.committed <= cursor <= self.delivered:
            raise ValueError("commit must refer to delivered samples")
        self.committed = cursor

    def state_dict(self) -> dict:
        return {"sampler": self.sampler.identity, "cursor": self.committed}

    def metrics(self) -> dict:
        with self.reader.lock:
            requested, read, calls = self.reader.requested_bytes, self.reader.read_bytes, self.reader.read_calls
            slowest_read = asdict(self.reader.slowest_read) if self.reader.slowest_read is not None else None
        return {
            "loader_mode": self.mode,
            "startup_seconds": self.startup_seconds,
            "preload_seconds": self.preload_seconds,
            "startup_read_bytes": self.startup_read_bytes,
            "startup_read_calls": self.startup_read_calls,
            "resident_token_bytes": self.resident_tokens.nbytes,
            "resident_index_bytes": self.resident_indices.nbytes,
            "resident_loaded_blocks": self.resident_loaded_blocks,
            "resident_total_blocks": len(self.resident_indices),
            "resident_complete": self.resident_ready.is_set(),
            "preload_requested_bytes": requested if self.mode == "resident" else 0,
            "preload_read_bytes": read if self.mode == "resident" else 0,
            "preload_read_calls": calls if self.mode == "resident" else 0,
            "training_read_bytes": read - self.startup_read_bytes if self.ready.is_set() else 0,
            "training_read_calls": calls - self.startup_read_calls if self.ready.is_set() else 0,
            "sampler_index_bytes": self.sampler.index_bytes,
            "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            "requested_bytes": requested,
            "read_bytes": read,
            "read_calls": calls,
            "slowest_read": slowest_read,
            "read_amplification": read / max(requested, 1),
            "batches": self.batches,
            "queue_starvations": self.starved,
            "queue_occupancy_mean": float(np.mean(self.occupancies)) if self.occupancies else 0,
            "wait_p50_ms": float(np.percentile(self.waits, 50)) * 1000 if self.waits else 0,
            "wait_p95_ms": float(np.percentile(self.waits, 95)) * 1000 if self.waits else 0,
            "wait_p99_ms": float(np.percentile(self.waits, 99)) * 1000 if self.waits else 0,
            "wait_max_ms": max(self.waits, default=0) * 1000,
            "wait_lifetime_max_ms": self.largest_waits[0]["wait_ms"] if self.largest_waits else 0,
            "wait_lifetime_total_ms": self.wait_total * 1000,
            "largest_waits": list(self.largest_waits),
            "wait_window_batches": len(self.waits),
            "memory_estimate_bytes": self.memory,
        }

    def close(self) -> None:
        self.cancel.set()
        self.changed.set()
        self.thread.join()
        self.pool.shutdown(wait=True, cancel_futures=True)
        self.reader.close()
        self.sampler.close()
        self.resident_indices = np.empty(0, dtype=np.int64)
        self.resident_remaining = np.empty(0, dtype=np.int64)
        self.resident_tokens = np.empty((0, self.config.model.context_length + 1), dtype="<u2")
        self.specs = iter(())
        while not self.items.empty():
            self.items.get_nowait()

    def __enter__(self) -> TokenLoader:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


class BatchSource(Protocol):
    config: Config

    def next(self) -> LoadedBatch: ...


class DevicePrefetcher:
    """Double-buffered pinned inputs; H2D completion events guard host buffer reuse."""

    def __init__(self, loader: BatchSource, device: torch.device):
        self.loader, self.device = loader, device
        self.stream = torch.cuda.Stream(device=device) if device.type == "cuda" else None
        shape = (loader.config.training.micro_batch_size, loader.config.model.context_length)
        self.buffers = [
            tuple(torch.empty(shape, dtype=torch.int64, pin_memory=self.stream is not None) for _ in range(2))
            for _ in range(2)
        ]
        self.events: list[torch.cuda.Event | None] = [None, None]
        self.slot = 0
        self.pending: tuple[Batch, BatchSpec] | None = None
        self.exhausted = False

    def preload(self) -> None:
        if self.pending is not None or self.exhausted:
            return
        try:
            loaded = self.loader.next()
        except StopIteration:
            self.exhausted = True
            return
        slot = self.slot
        self.slot = 1 - slot
        event = self.events[slot]
        if event is not None:
            event.synchronize()
        # Empty ranks still participate in the final DDP backward with a zero loss.
        count = max(1, len(loaded.tokens))
        x, y = (buffer[:count] for buffer in self.buffers[slot])
        if len(loaded.tokens):
            x.copy_(torch.from_numpy(loaded.tokens[:, :-1]))
            y.copy_(torch.from_numpy(loaded.tokens[:, 1:]))
        else:
            x.zero_()
            y.fill_(-100)
        if self.stream is not None:
            with torch.cuda.stream(self.stream):
                batch = (x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True))
                event = torch.cuda.Event()
                event.record(self.stream)
                self.events[slot] = event
        else:
            # CPU tensors cannot alias a slot overwritten by the next preload.
            batch = (x.clone(), y.clone())
        self.pending = (batch, loaded.spec)

    def next(self) -> tuple[Batch, BatchSpec]:
        self.preload()
        if self.pending is None:
            raise StopIteration
        batch, spec = self.pending
        if self.stream is not None:
            current = torch.cuda.current_stream(self.device)
            current.wait_stream(self.stream)
            for tensor in batch:
                tensor.record_stream(current)
        self.pending = None
        return batch, spec

    def close(self) -> None:
        if self.stream is not None:
            self.stream.synchronize()
