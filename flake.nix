{
  description = "A basic Nix flake providing development shells";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.11";
    nixpkgs-stable.follows = "nixpkgs";
    nixpkgs-unstable.url = "github:nixos/nixpkgs?ref=nixos-unstable";
    nixpkgs-v2511.url = "github:NixOS/nixpkgs/nixos-25.11";
    nixpkgs-v2505.url = "github:NixOS/nixpkgs/nixos-25.05";
    nixpkgs-v2411.url = "github:NixOS/nixpkgs/nixos-24.11";
    nur = {
      url = "github:nix-community/NUR";
    };
  };

  outputs =
    {
      self,
      nixpkgs,
      nixpkgs-stable,
      nixpkgs-unstable,
      nixpkgs-v2511,
      nixpkgs-v2505,
      nixpkgs-v2411,
      nur,
      ...
    }@inputs:
    let
      pkg-settings = rec {
        allowed-unfree-packages =
          pkg:
          builtins.elem (nixpkgs.lib.getName pkg) [
            "cudnn"
            "libcublas"
            "cuda_nvrtc"
            "cuda_cudart"
            "cuda_nvcc"
            "cuda_cccl"
            "cudatoolkit"
            "nvidia-driver"
            "cuda-toolkit"
            "cuda-stubs"
          ];
        allowed-insecure-packages = [
          "electron-11.5.0"
          "openssl-1.1.1w"
        ];
      };

      eachSystem = nixpkgs.lib.genAttrs [ "x86_64-linux" ] (
        system:
        let
          pkgs = import nixpkgs {
            inherit system;
            config.allowUnfreePredicate = pkg-settings.allowed-unfree-packages;
            config.permittedInsecurePackages = pkg-settings.allowed-insecure-packages;
            overlays = [
              nur.overlays.default
              (final: prev: {
                unstable = import nixpkgs-unstable {
                  inherit system;
                  config.allowUnfreePredicate = pkg-settings.allowed-unfree-packages;
                  config.permittedInsecurePackages = pkg-settings.allowed-insecure-packages;
                  overlays = [ nur.overlays.default ];
                };
              })
              (final: prev: {
                v2511 = import nixpkgs-v2511 {
                  inherit system;
                  config.allowUnfreePredicate = pkg-settings.allowed-unfree-packages;
                  config.permittedInsecurePackages = pkg-settings.allowed-insecure-packages;
                  overlays = [ nur.overlays.default ];
                };
              })
              (final: prev: {
                v2505 = import nixpkgs-v2505 {
                  inherit system;
                  config.allowUnfreePredicate = pkg-settings.allowed-unfree-packages;
                  config.permittedInsecurePackages = pkg-settings.allowed-insecure-packages;
                  overlays = [ nur.overlays.default ];
                };
              })
              (final: prev: {
                v2411 = import nixpkgs-v2411 {
                  inherit system;
                  config.allowUnfreePredicate = pkg-settings.allowed-unfree-packages;
                  config.permittedInsecurePackages = pkg-settings.allowed-insecure-packages;
                  overlays = [ nur.overlays.default ];
                };
              })
            ];
          };

          withPkgs = pkgs: {
            # 构建工具
            buildTools = with pkgs; [
              pkg-config
              bear
              gnumake
              ninja
              cmake
              xmake
            ];

            # 编译器和调试工具
            compilers = with pkgs; [
              llvmPackages_20.clangNoLibcxx
              gcc15
              gdb
            ];

            # C++ 库
            cppLibs = with pkgs; [
              boost
              spdlog
              fmt
              cli11
              cpptrace
              gtest
              gbenchmark
            ];

            # 系统库
            systemLibs = with pkgs; [
              stdenv.cc.cc.lib
              libz
              openssl
              cacert
              bzip2
              zlib
              zip
              libdwarf
              glib
              pcre
              libffi
              zstd
              xz
              zlib
              zstd
              bzip2
              expat
              libiconv
              dbus
              krb5
              brotli
              libpulseaudio
            ];

            # 图形/数学库
            graphicsLibs = with pkgs; [
              assimp
              eigen
              glew
              glfw
              glm
              libGL
            ];

            # Wayland 库（Qt wayland 后端需要）
            waylandLibs = with pkgs; [
              wayland
              wayland-protocols
              libdrm
              wayland-scanner
            ];

            xorgLibs = with pkgs; [
              xorg.libX11
              xorg.libXrandr
              xorg.libXinerama
              xorg.libXi
              xorg.libXxf86vm
              xorg.libXcursor
              libxkbcommon
              xorg.xorgproto
              xorg.libxcb
              xorg.libXext
              xorg.libXfixes
              xorg.libXrender
              xorg.libXcomposite
              xorg.libXdamage
              xorg.libXres
            ];

            # CUDA 运行时库（PyTorch CUDA 需要）
            cudaLibs = with pkgs; [
              # CUDA 基础库
              cudaPackages_12.cuda_nvrtc
              cudaPackages_12.cuda_cudart
              # cuDNN 和 cuBLAS
              cudaPackages_12.cudnn
              cudaPackages_12.libcublas
            ];

            # SDL 库
            sdlLibs = with pkgs; [
              SDL2
              SDL2_gfx
              SDL2_net
              SDL2_mixer
              SDL2_ttf
              SDL2_sound
              SDL2_image
              SDL2_Pango
              sdl3
              sdl3-image
              sdl3-ttf
            ];

            # Qt 库
            qtLibs = with pkgs; [
              qt6.qtbase
              qt6.qtmultimedia
              qt6.qtdeclarative
              qt6.qttools
              qt6.qtnetworkauth
              qt6.qtwebchannel
              qt6.qtpositioning
              qt6.qt5compat
              qt6.qtsensors
              qt6.qtserialport
              qt6.qtremoteobjects
              qt6.qtimageformats
              qt6.qtsvg
              qt6.qtscxml
              qt6.qtwayland
            ];

            # GTK 库
            gtkLibs = with pkgs; [
              gtk2
              gtk3
              gtk4
            ];

            # 媒体库
            mediaLibs = with pkgs; [
              stb
              (opencv.override {
                enableFfmpeg = true;
                enablePython = false;
                enableContrib = true;
              })
              ffmpeg_7-full
              fontconfig
              freetype
              dav1d
              libaom
              libglibutil
              flac
            ];

            pythonEnv = with pkgs; [
              v2511.python312
              v2511.python312Packages.uv
              v2511.python312Packages.opencv4Full
            ];
          };

          pkgSets = withPkgs pkgs;

          allLibraries = pkgs.lib.flatten [
            pkgSets.systemLibs
            pkgSets.graphicsLibs
            pkgSets.waylandLibs
            pkgSets.xorgLibs
            pkgSets.mediaLibs
            # pkgSets.gtkLibs
            # pkgSets.buildTools
            # pkgSets.compilers
            # pkgSets.cppLibs
            # pkgSets.cudaLibs
            # pkgSets.sdlLibs
            # pkgSets.qtLibs
          ];

        in
        {
          default = pkgs.mkShellNoCC {
            name = "base";
            hardeningDisable = [ "fortify" ];

            packages = pkgs.lib.flatten [
              allLibraries
              pkgSets.pythonEnv
            ];

            shellHook = ''
              # NVIDIA 驱动库（PyTorch 需要 libcuda.so.1）
              export LD_LIBRARY_PATH=/run/opengl-driver/lib:$LD_LIBRARY_PATH
              # 添加系统 opencv4Full 到 PYTHONPATH（优先于 venv，有 AV1 支持）
              export PYTHONPATH=${pkgs.v2511.python312Packages.opencv4Full}/lib/python3.12/site-packages:$PYTHONPATH
              # 防止 PySide6 加载系统的 Qt 插件（libqgtk3 会引入 GTK 文件对话框并崩溃）
              unset QT_PLUGIN_PATH
            '';

            env.LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath allLibraries;
          };
        }
      );

    in
    {
      devShells = eachSystem;
    };
}
