{ pkgs, lib, config, inputs, ... }:
let
  pkgs-unstable = import inputs.nixpkgs-unstable { system = pkgs.stdenv.system; };
in 
{
  packages = with pkgs-unstable; [
    basedpyright
    cmake
    gfortran11
    pre-commit
    ruff
    zlib
  ];

  languages.python = {
    enable = true;
    version = "3.11";
    manylinux.enable = false;
    venv.enable = true;
    uv = {
      enable = true;
      package = pkgs-unstable.uv;
    };
  };

  enterShell = ''
    python --version
  '';

  enterTest = ''
    pytest
  '';
}
