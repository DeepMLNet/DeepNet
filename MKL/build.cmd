SET VCPATH=C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC
SET MKLROOT=C:\Program Files (x86)\IntelSWTools\compilers_and_libraries\windows\mkl

CALL "%VCPATH%\vcvarsall.bat" amd64
CD %~dp0
IF EXIST build RD /S /Q build
MKDIR build
PUSHD build
XCOPY "%MKLROOT%\tools\builder\makefile" .
XCOPY /E "%MKLROOT%\tools\builder\lib" lib\

nmake libintel64 export=..\funcs.txt name=..\tensor_mkl manifest=no "MKLROOT=%MKLROOT%"
IF NOT ERRORLEVEL 0 GOTO eof

POPD
RD /S /Q build
DEL tensor_mkl.exp
DEL tensor_mkl.lib

:eof



