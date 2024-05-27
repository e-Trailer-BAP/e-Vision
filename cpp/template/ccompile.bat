rmdir /s /q "./build"
timeout /t 1
call "./configure.bat"
call "./build.bat"
call "./run.bat"