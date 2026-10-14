#!/usr/bin/env bash
# Slingshot/CXI transport activation; source on an allocated GH200 node.
# These installation paths match the measured stack; adapt them for another site.
# Eight-rank collective checks and training benchmarks passed; see the performance
# report for transport measurements and separate kernel numerical limitations.
# Request sbatch --network=job_vni. This script does not replace PyTorch's NCCL.

activate_cxi() {
    if [[ "$(uname -m)" != aarch64 ]]; then
        printf 'CXI activation requires an aarch64 compute node.\n' >&2
        return 1
    fi
    local plugin_root=/software/sse2/el9_gh200/manual/AWS_OFI_NCCL/1.21.1/g14/cu13.0/cmp91/nccl2.31.2/hpc1
    local fabric_root=/software/sse2/el9_gh200/manual/libfabric/2.6.0/g11/libcxi1.8.0/cu13.0
    local gcc_root=/software/sse2/el9_gh200/manual/GCC/14.3.0/hpc1
    local gdrcopy_root=/software/sse2/el9_gh200/easybuild/prefix/software/GDRCopy/2.5-GCCcore-14.3.0
    local library
    for library in \
        "$plugin_root/lib/libnccl-net-ofi.so" \
        "$fabric_root/lib/libfabric.so.1" \
        "$gcc_root/lib64/libstdc++.so.6" \
        "$gdrcopy_root/lib/libgdrapi.so"; do
        if [[ ! -r "$library" ]]; then
            printf 'Required CXI transport library is not readable: %s\n' "$library" >&2
            return 1
        fi
    done
    # Select dependencies before Python imports torch/libstdc++; the plugin needs
    # GLIBCXX_3.4.32. Its embedded RPATH does not select a newer libfabric.
    export LD_LIBRARY_PATH="$fabric_root/lib:$gcc_root/lib64:$gdrcopy_root/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export NCCL_NET_PLUGIN="$plugin_root/lib/libnccl-net-ofi.so"
    export NCCL_NET='AWS Libfabric'
    export NCCL_SOCKET_IFNAME=hsn
    export NCCL_LAUNCH_ORDER_IMPLICIT=1
    export FI_PROVIDER=cxi
    export OFI_NCCL_PROTOCOL=RDMA
    export FI_HMEM_CUDA_USE_DMABUF=1
    export FI_CXI_ENABLE_WRITEDATA=1
}

activate_cxi
