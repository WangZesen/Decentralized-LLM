"""Bounded positional network reads with ordered delivery and asynchronous transfer."""

import os
import queue
import resource
import threading
import time
from bisect import bisect_right
from collections import OrderedDict, deque
from concurrent.futures import Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Protocol

import numpy as np
import torch

from decentralized_llm.cache import TokenCache
from decentralized_llm.config import Config
from decentralized_llm.sampling import BatchSpec, GlobalSampler, check_memory
from decentralized_llm.state import Batch, TokenArray


@dataclass
class LoadedBatch:
    spec: BatchSpec
    tokens: TokenArray


@dataclass
class Segment:
    path: Path
    offset: int
    size: int
    destination: TokenArray
    destination_offset: int
    sample: int


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
        try:
            descriptor = self.descriptor(path)
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
            for segment in segments:
                relative = segment.offset - offset
                values = np.frombuffer(data, dtype="<u2", count=segment.size // 2, offset=relative)
                start = segment.destination_offset
                segment.destination[start : start + len(values)] = values
            with self.lock:
                self.requested_bytes += sum(s.size for s in segments)
        except ReadCancelled:
            raise
        except Exception as error:
            samples = [s.sample for s in segments[:8]]
            raise OSError(f"read failed: shard={path}, samples={samples}, byte_offset={offset}: {error}") from error

    def groups(self, segments: list[Segment]) -> list[list[Segment]]:
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
                and size <= self.config.max_read_amplification * (requested + segment.size)
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
        self.memory = check_memory(cache, config)
        self.cache, self.config = cache, config
        self.sampler = GlobalSampler(cache, config.model.context_length, config.runtime.seed)
        self.committed = cursor
        self.delivered = cursor
        self.specs = self.sampler.batches(config, rank, cursor, stop)
        self.cancel = threading.Event()
        self.ready = threading.Event()
        self.slots = threading.BoundedSemaphore(config.data.prefetch_batches)
        self.items: queue.Queue[LoadedBatch | BaseException | None] = queue.Queue(config.data.prefetch_batches)
        self.reader = PositionalReader(config, self.cancel)
        self.pool = ThreadPoolExecutor(max_workers=config.data.reader_threads, thread_name_prefix="token-read")
        self.waits: deque[float] = deque(maxlen=8192)
        self.occupancies: deque[int] = deque(maxlen=8192)
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

    def _segments(self, batches: list[LoadedBatch]) -> list[Segment]:
        segments: list[Segment] = []
        length = self.config.model.context_length
        offsets, paths = self.cache.offsets["train"], self.cache.paths["train"]
        max_tokens = self.config.data.read_range_bytes // 2
        for batch in batches:
            for row, sample in enumerate(batch.spec.indices):
                start = int(sample) * length
                end = start + length + 1
                dest = 0
                while start < end:
                    shard = bisect_right(offsets, start) - 1
                    stop = min(end, offsets[shard + 1], start + max_tokens)
                    segments.append(
                        Segment(
                            paths[shard],
                            (start - offsets[shard]) * 2,
                            (stop - start) * 2,
                            batch.tokens[row],
                            dest,
                            int(sample),
                        )
                    )
                    dest += stop - start
                    start = stop
        return segments

    def _produce(self) -> None:
        pending: list[Future[None]] = []
        try:
            while not self.cancel.is_set():
                # Reserve queue + in-flight capacity together: at most depth microbatches
                # are outstanding, even while the consumer is using a delivered batch.
                while not self.cancel.is_set():
                    if self.slots.acquire(timeout=0.1):
                        break
                else:
                    raise ReadCancelled()
                capacity = 1
                while capacity < self.config.data.prefetch_batches and self.slots.acquire(blocking=False):
                    capacity += 1
                specs = list(islice(self.specs, capacity))
                for _ in range(capacity - len(specs)):
                    self.slots.release()
                if not specs:
                    self.ready.set()
                    self._put(None)
                    return
                window = [
                    LoadedBatch(s, np.empty((len(s.indices), self.config.model.context_length + 1), dtype="<u2"))
                    for s in specs
                ]
                pending = [
                    self.pool.submit(self.reader.read_group, g) for g in self.reader.groups(self._segments(window))
                ]
                unfinished = set(pending)
                while unfinished:
                    done, unfinished = wait(unfinished, timeout=0.1)
                    for future in done:
                        future.result()
                    if self.cancel.is_set():
                        raise ReadCancelled()
                for batch in window:
                    if not self._put(batch):
                        return
                self.ready.set()
        except ReadCancelled:
            pass
        except BaseException as error:
            self._put(error)
        finally:
            for future in pending:
                future.cancel()
            self.ready.set()

    def warmup(self) -> None:
        self.ready.wait()

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
        self.occupancies.append(occupancy)
        self.waits.append(elapsed)
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
        return {
            "sampler_index_bytes": self.sampler.permutation.nbytes,
            "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            "requested_bytes": requested,
            "read_bytes": read,
            "read_calls": calls,
            "read_amplification": read / max(requested, 1),
            "batches": self.batches,
            "queue_starvations": self.starved,
            "queue_occupancy_mean": float(np.mean(self.occupancies)) if self.occupancies else 0,
            "wait_p50_ms": float(np.percentile(self.waits, 50)) * 1000 if self.waits else 0,
            "wait_p95_ms": float(np.percentile(self.waits, 95)) * 1000 if self.waits else 0,
            "wait_p99_ms": float(np.percentile(self.waits, 99)) * 1000 if self.waits else 0,
            "wait_max_ms": max(self.waits, default=0) * 1000,
            "wait_window_batches": len(self.waits),
            "memory_estimate_bytes": self.memory,
        }

    def close(self) -> None:
        self.cancel.set()
        self.thread.join()
        self.pool.shutdown(wait=True, cancel_futures=True)
        self.reader.close()
        self.sampler.permutation = np.empty(0, dtype=np.int64)
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
