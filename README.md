# MXTransform

MXTransform is a Rust program for doing matrix transformations on images.

## Usage

Run `mxtransform --help` to see the detailed usage information.

In general, you need to provide an input image with `-i` or `--input`, a transformation matrix with `-m` or `--matrix`, and an output image with `-o` or `--output`.

For example, to transform `input.png` to `output.png` using the following matrix, which will stretch the image horizontally by a factor of 2:

```text
| 2 0 |
| 0 1 |
```

You would run:

```sh
mxtransform -i input.png -o output.png -m 2,0,0,1
```

You can also specify an offset with `-f` or `--offset` followed by `X,Y`, in case the image ends up outside the bounds of the output image. For example to stretch the image horizontally by a factor of 2 and move the result 1920 pixels to the left:

```sh
mxtransform -i input.png -o output.png -m 2,0,0,1 -f -1920,0
```

## Installation

### Nix

1. Install Nix by following the instructions [on the official website](https://nixos.org/download.html).
2. Get MXTransform from the flake in this repository (you might need to [enable flake support](https://wiki.nixos.org/wiki/Flakes#Enable_flakes_temporarily)):

    ```sh
    nix shell github:GGORG0/mxtransform
    ```

3. You can now run MXTransform from the shell:

    ```sh
    mxtransform --help
    ```

### Cargo Binstall

Cargo Binstall downloads pre-built binaries from the GitHub releases page and installs them for you. You **don't need** to have Rust installed to use Cargo Binstall.

1. Install Cargo Binstall by following the instructions [in the official repository](https://github.com/cargo-bins/cargo-binstall?tab=readme-ov-file#installation). TL;DR:
    - **Linux/MacOS**

        ```sh
        curl -L --proto '=https' --tlsv1.2 -sSf https://raw.githubusercontent.com/cargo-bins/cargo-binstall/main/install-from-binstall-release.sh | bash
        ```

    - **MacOS**

        ```sh
        brew install cargo-binstall
        ```

    - **Windows (in PowerShell)**

        ```powershell
        Set-ExecutionPolicy Unrestricted -Scope Process; iex (iwr "https://raw.githubusercontent.com/cargo-bins/cargo-binstall/main/install-from-binstall-release.ps1").Content
        ```

2. **If you don't have Rust installed**, make sure to add the Cargo bin directory to your PATH.
    <details>
    The Binstall installer will tell you if you need to do this and conveniently print the path in question.
    - **Linux/MacOS**

        Temporarily:

        ```sh
        export PATH="$HOME/.cargo/bin:$PATH"
        ```

        Permanently:

        ```sh
        echo 'export PATH="$HOME/.cargo/bin:$PATH"' >> ~/.bashrc # or ~/.zshrc if you use zsh (default on MacOS)

        # Then restart your shell or run
        source ~/.bashrc # or ~/.zshrc
        ```

    - **Windows**: You'll need to modify the PATH environment variable manually.
        1. Open the Start Search, type in "env", and choose "Edit the system environment variables".
        2. Click the "Environment Variables..." button.
        3. Under "System Variables", find the PATH variable and click "Edit...".
        4. Add the path to the Cargo bin directory (e.g. `C:\Users\username\.cargo\bin` - the path that was printed by the Binstall installer) to the list of paths.
        5. Click "OK" on all the windows to close them.
        6. You may need to restart your shell, log out and back in, or restart your computer for the changes to take effect.
    </details>

3. Install MXTransform using Cargo Binstall:

    ```sh
    cargo-binstall mxtransform
    ```

4. You can now run MXTransform from anywhere in your shell:

    ```sh
    mxtransform --help
    ```

### Getting the binary from the GitHub releases page

1. Go to the [releases page](https://github.com/GGORG0/mxtransform/releases/latest).
2. Download the binary for your platform.
3. Extract the binary from the archive.
4. Open a shell in the directory where the binary is located.
5. Run the binary:

    - **Linux/MacOS**

        ```sh
        ./mxtransform --help
        ```

    - **Windows**

        ```cmd
        mxtransform.exe --help
        ```

### Cargo (compiling from crates.io)

1. Install Rust by using the official installer from [rustup.rs](https://rustup.rs/).
2. Install MXTransform from [crates.io](https://crates.io/crates/mxtransform) using Cargo:

    ```sh
    cargo install mxtransform
    ```

3. You can now run MXTransform from anywhere in your shell:

    ```sh
    mxtransform --help
    ```
