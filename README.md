# NyaZy

## Current `nyac` behavior

Read NyaZy source code from specified input file and compiles it into llvm ir.
```bash
# run the nyacc, generating sample.ll
$ ./bin nyac sample.nz
$ ./bin nyac sample.nz -o output.ll
# Specify output filename with debug output
$ ./bin nyac sample.nz -o output.ll -d --mlir-print-ir-after-all
# run the LLVM IR using LLVM's lli
$ ./bin lli sample.ll
```

## Setup

```bash
# build LLVM and other external libraries
$ ./bin thirdparty
# configure this project
$ ./bin configure
# build this project
$ ./bin build
```

## `bin` shell script
`/bin` can be used to invoke executables or one linear command for everyday developments.

Run `./bin help` to see all the available commands.
