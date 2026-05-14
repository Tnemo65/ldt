@echo off
REM =============================================================================
REM Test Coverage Report Generator for CA-DQStream + MemStream v5
REM Windows Batch Script
REM =============================================================================
REM
REM This script runs pytest with coverage reporting for the complete pipeline.
REM
REM Coverage targets:
REM   - operators/: Layer functions
REM   - core/: MemStream core + IEC
REM   - Minimum: 70% line coverage
REM
REM Usage:
REM   generate_coverage.bat          - Run with default settings
REM   generate_coverage.bat --html   - Generate HTML report
REM   generate_coverage.bat --xml    - Generate XML report (CI)
REM   generate_coverage.bat --no-fail - Don't fail on low coverage
REM
REM Reference: original_flow.md Section 8.8 (Test Matrix)

setlocal enabledelayedexpansion

set "PROJECT_ROOT=%~dp0.."
cd /d "%PROJECT_ROOT%"

set "GENERATE_HTML=false"
set "GENERATE_XML=false"
set "NO_FAIL=false"
set "COVERAGE_MIN=70"

REM Parse arguments
:parse_args
if "%~1"=="" goto :run_tests
if "%~1"=="--html" (
    set "GENERATE_HTML=true"
    shift
    goto :parse_args
)
if "%~1"=="--xml" (
    set "GENERATE_XML=true"
    shift
    goto :parse_args
)
if "%~1"=="--no-fail" (
    set "NO_FAIL=true"
    shift
    goto :parse_args
)
if "%~1"=="--min" (
    set "COVERAGE_MIN=%~2"
    shift
    shift
    goto :parse_args
)
echo Unknown option: %~1
echo Usage: %0 [--html] [--xml] [--no-fail] [--min COVERAGE_MIN]
exit /b 1

:run_tests
echo ================================================================================
echo CA-DQStream + MemStream v5 - Test Coverage Report
echo ================================================================================
echo.
echo Project root: %PROJECT_ROOT%
echo Coverage targets: operators, core
echo Minimum coverage: %COVERAGE_MIN%%%
echo.

REM Build pytest command
set "PYTEST_CMD=pytest tests/"

REM Check for pytest-cov
python -c "import pytest_cov" 2>nul
if errorlevel 1 (
    echo WARNING: pytest-cov not installed. Installing...
    pip install pytest-cov
)

REM Build coverage targets
set "COVERAGE_ARGS=--cov=operators --cov=core --cov-report=term -v --tb=short"

if "%GENERATE_HTML%"=="true" (
    set "COVERAGE_ARGS=!COVERAGE_ARGS! --cov-report=html"
    echo HTML coverage report will be generated in: htmlcov\index.html
)

if "%GENERATE_XML%"=="true" (
    set "COVERAGE_ARGS=!COVERAGE_ARGS! --cov-report=xml"
    echo XML coverage report will be generated in: coverage.xml
)

if "%NO_FAIL%"=="false" (
    set "COVERAGE_ARGS=!COVERAGE_ARGS! --cov-fail-under=%COVERAGE_MIN%"
)

echo Running tests with coverage...
echo.
echo Command: pytest tests/ !COVERAGE_ARGS!
echo.

REM Run tests
if "%NO_FAIL%"=="true" (
    python -m pytest tests/ !COVERAGE_ARGS! || echo Tests completed with some failures.
) else (
    python -m pytest tests/ !COVERAGE_ARGS!
)

echo.
echo ================================================================================
echo Coverage Report Generation Complete
echo ================================================================================
echo.

if "%GENERATE_HTML%"=="true" (
    echo HTML report: file://%PROJECT_ROOT%\htmlcov\index.html
)

if "%GENERATE_XML%"=="true" (
    echo XML report: %PROJECT_ROOT%\coverage.xml
)

echo.
echo To view HTML report:
echo   start htmlcov\index.html
echo.
echo To run specific test files:
echo   pytest tests\test_complete_pipeline.py -v
echo   pytest tests\test_integration.py -v
echo   pytest tests\test_memstream_core.py -v
echo.

endlocal
