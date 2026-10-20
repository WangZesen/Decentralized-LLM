"""Reciprocal logical-worker topology schedules."""


def peer_for(worker: int, size: int, topology: str, step: int) -> int:
    if size == 1:
        return worker
    if topology == "one_peer_ring":
        if size % 2:
            raise ValueError("ring requires even world size")
        return (worker + (1 if (worker + step % 2) % 2 == 0 else -1)) % size
    if topology == "one_peer_exponential":
        if size & (size - 1):
            raise ValueError("exponential requires power-of-two world size")
        return worker ^ (1 << (step % (size - 1).bit_length()))
    raise ValueError(f"unknown reciprocal topology: {topology}")


def validate_topology_size(topology: str, size: int) -> None:
    if size < 1:
        raise ValueError("world size must be positive")
    if topology == "one_peer_ring" and size > 1 and size % 2:
        raise ValueError("one_peer_ring requires an even world size (or 1)")
    if topology == "one_peer_exponential" and size & (size - 1):
        raise ValueError("one_peer_exponential requires a power-of-two world size")
    if topology not in ("complete", "one_peer_ring", "one_peer_exponential"):
        raise ValueError("unknown topology")
