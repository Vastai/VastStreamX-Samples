:: set windows envirenment

:: fix them 
set COMPILER_ROOT=C:\Program Files\AICompiler 1.3.0
set VASTSTREAM_ROOT=C:\Program Files\VastStream SDK
set VASTPIPE_ROOT=C:\Users\tanzh\Desktop\vastpipe-2.2.3
set SAMPLE_ROOT=C:\Users\tanzh\Desktop\projects\vastpipe-samples\build\vastpipe-samples

:: set compiler env
set PATH=%COMPILER_ROOT%\lib;%PATH%
:: set vaststream env
set PATH=%VASTSTREAM_ROOT%\bin;%PATH%
:: set vastpipe env
set PATH=%VASTPIPE_ROOT%\bin;%VASTPIPE_ROOT%\calculators;%VASTPIPE_ROOT%\x64\vc16\bin;%PATH%
set CMAKE_PREFIX_PATH=%VASTPIPE_ROOT%;%CMAKE_PREFIX_PATH%
:: set sample env
set PATH=%SAMPLE_ROOT%;%SAMPLE_ROOT%\vsx;%SAMPLE_ROOT%\vastpipe;%PATH%

