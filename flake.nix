{
  description = "NInfer development environment (CUDA sm_120a, C++20, conversion tools)";

  # CUDA 13.1 is the validated toolkit. On NixOS, keep this input in step with the
  # system nixpkgs so `nix develop` reuses store paths instead of building a second
  # CUDA stack: `nix develop --override-input nixpkgs nixpkgs`.
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = { self, nixpkgs }:
    let
      systems = [ "x86_64-linux" "aarch64-linux" ];
      forAll = f: nixpkgs.lib.genAttrs systems (system: f (import nixpkgs { inherit system; }));

      # CUDA, the NVIDIA driver and FFmpeg are unfree/non-redistributable, so the
      # shell cannot evaluate without allowing them. Naming them keeps the exception
      # narrow instead of setting allowUnfree globally.
      nvidiaNames = [
        "cuda_cudart" "cuda_nvcc" "cuda_cccl" "cuda_cupti" "cuda_nvrtc"
        "libcublas" "libcublasmp" "libcufft" "libcurand" "libcusolver"
        "libcusparse" "libcusparse_lt" "libnvjitlink" "libnpp" "libcufile"
        "cudatoolkit" "cuda-merged"
        "libcudnn" "cudnn" "nvidia-x11" "nvidia-kernel-module" "nvidia-settings" "nvidia-persistenced"
      ];
      allowUnfreePredicate = pkg: builtins.elem (nixpkgs.lib.getName pkg) nvidiaNames;

      mkShell = system:
        let
          pkgs = import nixpkgs {
            inherit system;
            config = {
              allowUnfreePredicate = allowUnfreePredicate;
              cudaSupport = true;
            };
          };
          cuda = pkgs.cudaPackages_13_1;

        in
        pkgs.mkShell {
          name = "ninfer";

          packages = (with pkgs; [
            cmake
            ninja
            pkg-config
            git
            ffmpeg
            curl
            zlib
            openssl
                      ]) ++ [
            cuda.cudatoolkit
          ];

          # The build rejects any architecture other than sm_120a. RTX 50 series
          # (GB20x: RTX 5060 through RTX 5090) is sm_120, so this one value covers
          # the whole family and no per-GPU switch is needed.
          CMAKE_CUDA_ARCHITECTURES = "120a";
          CUDA_PATH = cuda.cudatoolkit;
          CUDA_HOME = cuda.cudatoolkit;
          LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [ cuda.cudatoolkit ];

          shellHook = ''
            if [ -d /run/opengl-driver/lib ]; then
              export LD_LIBRARY_PATH="/run/opengl-driver/lib:$LD_LIBRARY_PATH"
            fi
            if [ -d /usr/lib/x86_64-linux-gnu ]; then
              export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"
            fi
            echo "NInfer dev shell"
            echo "  cmake  $(cmake --version | head -1 | cut -d' ' -f3)"
            echo "  nvcc   $(nvcc --version | tail -1)"
            echo "  python external (use the maintained test environment)"
            echo "  target sm_120a (RTX 50 series / GB20x)"
          '';
        };
    in
    {
      devShells = nixpkgs.lib.genAttrs systems (system: {
        default = mkShell system;
      });
    };
}
