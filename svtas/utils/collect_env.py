import sys
import cv2
import torch
import subprocess
import numpy as np
import os.path as osp
from collections import OrderedDict, defaultdict
from .package_utils import is_opencv_available, is_torchvision_available, get_git_hash

def is_rocm_pytorch() -> bool:
    """Check whether the PyTorch is compiled on ROCm."""
    is_rocm = False
    try:
        from torch.utils.cpp_extension import ROCM_HOME
        is_rocm = True if ((torch.version.hip is not None) and
                            (ROCM_HOME is not None)) else False
    except ImportError:
        pass
    return is_rocm

def _get_cuda_home():
    if is_rocm_pytorch():
        from torch.utils.cpp_extension import ROCM_HOME
        CUDA_HOME = ROCM_HOME
    else:
        from torch.utils.cpp_extension import CUDA_HOME
    return CUDA_HOME

def get_build_config():
    """Obtain the build information of PyTorch."""
    return torch.__config__.show()
    
def collect_basic_env():
    """Collect the information of the running environments.

    Returns:
        dict: The environment information. The following fields are contained.

            - sys.platform: The variable of ``sys.platform``.
            - Python: Python version.
            - CUDA available: Bool, indicating if CUDA is available.
            - GPU devices: Device type of each GPU.
            - CUDA_HOME (optional): The env var ``CUDA_HOME``.
            - NVCC (optional): NVCC version.
            - GCC: GCC version, "n/a" if GCC is not installed.
            - MSVC: Microsoft Virtual C++ Compiler version, Windows only.
            - PyTorch: PyTorch version.
            - PyTorch compiling details: The output of \
                ``torch.__config__.show()``.
            - TorchVision (optional): TorchVision version.
            - OpenCV (optional): OpenCV version.
    """
    from distutils import errors

    env_info = OrderedDict()
    env_info['sys.platform'] = sys.platform
    env_info['Python'] = sys.version.replace('\n', '')

    cuda_available = torch.cuda.is_available()
    env_info['CUDA available'] = cuda_available

    env_info['numpy_random_seed'] = np.random.get_state()[1][0]

    if cuda_available:
        devices = defaultdict(list)
        for k in range(torch.cuda.device_count()):
            devices[torch.cuda.get_device_name(k)].append(str(k))
        for name, device_ids in devices.items():
            env_info['GPU ' + ','.join(device_ids)] = name

        CUDA_HOME = _get_cuda_home()
        env_info['CUDA_HOME'] = CUDA_HOME

        if CUDA_HOME is not None and osp.isdir(CUDA_HOME):
            if CUDA_HOME == '/opt/rocm':
                try:
                    nvcc = osp.join(CUDA_HOME, 'hip/bin/hipcc')
                    nvcc = subprocess.check_output(
                        f'"{nvcc}" --version', shell=True)
                    nvcc = nvcc.decode('utf-8').strip()
                    release = nvcc.rfind('HIP version:')
                    build = nvcc.rfind('')
                    nvcc = nvcc[release:build].strip()
                except subprocess.SubprocessError:
                    nvcc = 'Not Available'
            else:
                try:
                    nvcc = osp.join(CUDA_HOME, 'bin/nvcc')
                    nvcc = subprocess.check_output(f'"{nvcc}" -V', shell=True)
                    nvcc = nvcc.decode('utf-8').strip()
                    release = nvcc.rfind('Cuda compilation tools')
                    build = nvcc.rfind('Build ')
                    nvcc = nvcc[release:build].strip()
                except subprocess.SubprocessError:
                    nvcc = 'Not Available'
            env_info['NVCC'] = nvcc

    try:
        # Check C++ Compiler.
        # For Unix-like, sysconfig has 'CC' variable like 'gcc -pthread ...',
        # indicating the compiler used, we use this to get the compiler name
        import io
        import sysconfig
        cc = sysconfig.get_config_var('CC')
        if cc:
            cc = osp.basename(cc.split()[0])
            cc_info = subprocess.check_output(f'{cc} --version', shell=True)
            env_info['GCC'] = cc_info.decode('utf-8').partition(
                '\n')[0].strip()
        else:
            # on Windows, cl.exe is not in PATH. We need to find the path.
            # distutils.ccompiler.new_compiler() returns a msvccompiler
            # object and after initialization, path to cl.exe is found.
            import locale
            import os
            from distutils.ccompiler import new_compiler
            ccompiler = new_compiler()
            ccompiler.initialize()
            cc = subprocess.check_output(
                f'{ccompiler.cc}', stderr=subprocess.STDOUT, shell=True)
            encoding = os.device_encoding(
                sys.stdout.fileno()) or locale.getpreferredencoding()
            env_info['MSVC'] = cc.decode(encoding).partition('\n')[0].strip()
            env_info['GCC'] = 'n/a'
    except (subprocess.CalledProcessError, errors.DistutilsPlatformError):
        env_info['GCC'] = 'n/a'
    except io.UnsupportedOperation as e:
        # JupyterLab on Windows changes sys.stdout, which has no `fileno` attr
        # Refer to: https://github.com/open-mmlab/mmengine/issues/931
        # TODO: find a solution to get compiler info in Windows JupyterLab,
        # while preserving backward-compatibility in other systems.
        env_info['MSVC'] = f'n/a, reason: {str(e)}'

    env_info['PyTorch'] = torch.__version__
    env_info['PyTorch compiling details'] = get_build_config()

    if is_torchvision_available():
        import torchvision
        env_info['TorchVision'] = torchvision.__version__

    if is_opencv_available():
        env_info['OpenCV'] = cv2.__version__

    return env_info

def collect_env():
    env_info = collect_basic_env()
    env_info['SVTAS'] = (get_git_hash(digits=7))
    return env_info


if __name__ == '__main__':
    for name, val in collect_env().items():
        print(f'{name}: {val}')