@echo off
REM Build the GDBot Bridge mod. Needs VS 2022+ (vcvars64), clang, CMake+Ninja and
REM the Geode CLI. Adjust the paths below if your toolchain lives elsewhere.
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
set "GEODE_SDK=C:\Users\danie\.geode-sdk"
set "PATH=%PATH%;C:\Program Files\LLVM\bin;C:\Users\danie\.geode-cli;C:\Program Files\Microsoft Visual Studio\18\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin;C:\Program Files\Microsoft Visual Studio\18\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja"
cd /d "%~dp0"
echo === geode build ===
geode build %*
echo === EXITCODE %ERRORLEVEL% ===
