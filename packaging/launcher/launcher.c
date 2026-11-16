/*
 * UtauCover launcher (macOS and Windows).
 *
 * Runs the Python that ships inside the app, in-process, so the running program *is* UtauCover:
 * the Dock / taskbar show its name and icon, not "Python". The interpreter is loaded at run time
 * (dlopen / LoadLibrary) from the bundled runtime folder, so nothing outside the app is needed.
 *
 *   macOS:   UtauCover.app/Contents/MacOS/<this>   runtime: Contents/Resources/runtime
 *   Windows: UtauCover\<this>.exe                   runtime: UtauCover\runtime
 *
 * Built twice: UC_GUI=1 opens the window (utaucover.gui); UC_GUI=0 is the command line
 * ("utaucover-cli build song.mp3 ...", "utaucover-cli mcp").
 */
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef UC_GUI
#define UC_GUI 0
#endif
#define UC_MODULE (UC_GUI ? L"utaucover.gui" : L"utaucover")

typedef void (*PFN_PreInitIsolated)(PyPreConfig *);
typedef PyStatus (*PFN_PreInitialize)(const PyPreConfig *);
typedef void (*PFN_InitIsolated)(PyConfig *);
typedef PyStatus (*PFN_SetString)(PyConfig *, wchar_t **, const wchar_t *);
typedef PyStatus (*PFN_InitFromConfig)(const PyConfig *);
typedef int (*PFN_StatusException)(PyStatus);
typedef void (*PFN_ConfigClear)(PyConfig *);
typedef int (*PFN_RunMain)(void);
#ifdef _WIN32
typedef PyStatus (*PFN_SetArgv)(PyConfig *, Py_ssize_t, wchar_t *const *);
#else
typedef PyStatus (*PFN_SetBytesArgv)(PyConfig *, Py_ssize_t, char *const *);
typedef PyStatus (*PFN_SetBytesString)(PyConfig *, wchar_t **, const char *);
#endif

#ifdef _WIN32
/* ------------------------------------------------------------------------------------- Windows */
#include <windows.h>
#include <shellapi.h>

static void fail(const wchar_t *what, const wchar_t *detail) {
    wchar_t msg[2048];
    _snwprintf(msg, 2047, L"UtauCover can't start: %ls\n\n%ls\n\nTry extracting or installing UtauCover again.",
               what, detail ? detail : L"");
    msg[2047] = 0;
#if UC_GUI
    MessageBoxW(NULL, msg, L"UtauCover", MB_ICONERROR | MB_OK);
#else
    fwprintf(stderr, L"%ls\n", msg);
#endif
    exit(1);
}

static void fail_narrow(const wchar_t *what, const char *detail) {
    wchar_t wide[1024] = L"";
    if (detail) MultiByteToWideChar(CP_UTF8, 0, detail, -1, wide, 1023);
    fail(what, wide);
}

static void fail_status(PyStatus s) { fail_narrow(L"Python failed to initialise.", s.err_msg); }

/* plain main / WinMain (arguments come from GetCommandLineW), so no -municode is needed */
#if UC_GUI
int WINAPI WinMain(HINSTANCE inst, HINSTANCE prev, LPSTR cmdline, int show) {
#else
int main(void) {
#endif
    int argc = 0;
    wchar_t **argv = CommandLineToArgvW(GetCommandLineW(), &argc);
    wchar_t exe[MAX_PATH * 4], root[MAX_PATH * 4], runtime[MAX_PATH * 4], dll[MAX_PATH * 4], python[MAX_PATH * 4];
    DWORD n = GetModuleFileNameW(NULL, exe, MAX_PATH * 4);
    if (n == 0 || n >= MAX_PATH * 4) fail(L"Can't find where UtauCover is.", NULL);
    wcscpy(root, exe);
    wchar_t *slash = wcsrchr(root, L'\\');
    if (slash) *slash = 0;
    _snwprintf(runtime, MAX_PATH * 4, L"%ls\\runtime", root);
    _snwprintf(dll, MAX_PATH * 4, L"%ls\\python3%d.dll", runtime, PY_MINOR_VERSION);
    _snwprintf(python, MAX_PATH * 4, L"%ls\\python.exe", runtime);

    /* children (GAME helper, ffmpeg) get a clean environment pointing back at this install */
    SetEnvironmentVariableW(L"UTAUCOVER_APP_DIR", root);
    SetEnvironmentVariableW(L"PYTHONHOME", NULL);
    SetEnvironmentVariableW(L"PYTHONPATH", NULL);
    SetEnvironmentVariableW(L"PYTHONUTF8", L"1");
    /* group the window under the Start-menu shortcut (same id as the installer's shortcut) */
    {
        typedef HRESULT (WINAPI *PFN_SetAppId)(PCWSTR);
        HMODULE sh = LoadLibraryW(L"shell32.dll");
        PFN_SetAppId set = sh ? (PFN_SetAppId)GetProcAddress(sh, "SetCurrentProcessExplicitAppUserModelID") : NULL;
        if (set) set(L"UtauCover.UtauCover");
    }

    SetDllDirectoryW(runtime);
    HMODULE py = LoadLibraryExW(dll, NULL, LOAD_WITH_ALTERED_SEARCH_PATH);
    SetDllDirectoryW(NULL);
    if (!py) fail(L"The bundled Python is missing or damaged.", dll);
#define SYM(T, name) T p_##name = (T)GetProcAddress(py, #name); if (!p_##name) fail_narrow(L"The bundled Python is incomplete.", #name)
#else
/* ------------------------------------------------------------------------------- macOS / POSIX */
#include <CoreFoundation/CoreFoundation.h>
#include <dlfcn.h>
#include <libgen.h>
#include <limits.h>
#include <mach-o/dyld.h>

static void fail(const char *what, const char *detail) {
    fprintf(stderr, "UtauCover can't start: %s\n%s\n", what, detail ? detail : "");
#if UC_GUI
    /* a Finder launch has no terminal: say it in a dialog */
    char msg[2048];
    snprintf(msg, sizeof msg, "%s\n\n%s\n\nTry downloading UtauCover again.", what, detail ? detail : "");
    CFStringRef m = CFStringCreateWithCString(NULL, msg, kCFStringEncodingUTF8);
    CFOptionFlags response;
    CFUserNotificationDisplayAlert(0, kCFUserNotificationStopAlertLevel, NULL, NULL, NULL,
                                   CFSTR("UtauCover can't start"), m, NULL, NULL, NULL, &response);
    if (m) CFRelease(m);
#endif
    exit(1);
}

static void fail_status(PyStatus s) { fail("Python failed to initialise.", s.err_msg); }

int main(int argc, char **argv) {
    char exe_raw[PATH_MAX], exe[PATH_MAX], root[PATH_MAX], runtime[PATH_MAX], lib[PATH_MAX], python[PATH_MAX];
    uint32_t size = sizeof exe_raw;
    if (_NSGetExecutablePath(exe_raw, &size) != 0 || !realpath(exe_raw, exe)) fail("Can't find where UtauCover is.", NULL);
    char dir[PATH_MAX];
    strncpy(dir, exe, sizeof dir - 1);
    dir[sizeof dir - 1] = 0;
    /* <app>/Contents/MacOS/<exe>  ->  <app>/Contents/Resources */
    snprintf(root, sizeof root, "%s/../Resources", dirname(dir));
    char resolved[PATH_MAX];
    if (realpath(root, resolved)) strncpy(root, resolved, sizeof root - 1);
    snprintf(runtime, sizeof runtime, "%s/runtime", root);
    snprintf(lib, sizeof lib, "%s/lib/libpython3.%d.dylib", runtime, PY_MINOR_VERSION);
    snprintf(python, sizeof python, "%s/bin/python3.%d", runtime, PY_MINOR_VERSION);

    setenv("UTAUCOVER_APP_DIR", root, 1);
    unsetenv("PYTHONHOME");
    unsetenv("PYTHONPATH");
    setenv("PYTHONUTF8", "1", 1);

    void *py = dlopen(lib, RTLD_NOW | RTLD_GLOBAL);
    if (!py) fail("The bundled Python is missing or damaged.", dlerror());
#define SYM(T, name) T p_##name = (T)dlsym(py, #name); if (!p_##name) fail("The bundled Python is incomplete.", #name)
#endif

    SYM(PFN_PreInitIsolated, PyPreConfig_InitIsolatedConfig);
    SYM(PFN_PreInitialize, Py_PreInitialize);
    SYM(PFN_InitIsolated, PyConfig_InitIsolatedConfig);
    SYM(PFN_SetString, PyConfig_SetString);
    SYM(PFN_InitFromConfig, Py_InitializeFromConfig);
    SYM(PFN_StatusException, PyStatus_Exception);
    SYM(PFN_ConfigClear, PyConfig_Clear);
    SYM(PFN_RunMain, Py_RunMain);

    /* isolated: ignore PYTHON* variables and the user's site-packages; UTF-8 everywhere */
    PyPreConfig pre;
    p_PyPreConfig_InitIsolatedConfig(&pre);
    pre.utf8_mode = 1;
    PyStatus st = p_Py_PreInitialize(&pre);
    if (p_PyStatus_Exception(st)) fail_status(st);

    PyConfig config;
    p_PyConfig_InitIsolatedConfig(&config);
    config.write_bytecode = 0;      /* the program folder may be read-only; .pyc files are precompiled */
    config.parse_argv = 0;          /* user arguments go to UtauCover untouched */
    config.install_signal_handlers = 1;
#ifdef _WIN32
    SYM(PFN_SetArgv, PyConfig_SetArgv);
    st = p_PyConfig_SetString(&config, &config.home, runtime);
    if (p_PyStatus_Exception(st)) fail_status(st);
    /* sys.executable = the real interpreter, so helper processes start Python, not another window */
    st = p_PyConfig_SetString(&config, &config.executable, python);
    if (p_PyStatus_Exception(st)) fail_status(st);
    st = p_PyConfig_SetArgv(&config, argc, argv);
#else
    SYM(PFN_SetBytesArgv, PyConfig_SetBytesArgv);
    SYM(PFN_SetBytesString, PyConfig_SetBytesString);
    st = p_PyConfig_SetBytesString(&config, &config.home, runtime);
    if (p_PyStatus_Exception(st)) fail_status(st);
    st = p_PyConfig_SetBytesString(&config, &config.executable, python);
    if (p_PyStatus_Exception(st)) fail_status(st);
    /* Finder adds "-psn_..." when launching an app: drop it */
    int keep = 0;
    for (int i = 0; i < argc; i++)
        if (i == 0 || strncmp(argv[i], "-psn_", 5) != 0) argv[keep++] = argv[i];
    argc = keep;
    st = p_PyConfig_SetBytesArgv(&config, argc, argv);
#endif
    if (p_PyStatus_Exception(st)) fail_status(st);
    st = p_PyConfig_SetString(&config, &config.run_module, UC_MODULE);
    if (p_PyStatus_Exception(st)) fail_status(st);

    st = p_Py_InitializeFromConfig(&config);
    p_PyConfig_Clear(&config);
    if (p_PyStatus_Exception(st)) fail_status(st);
    return p_Py_RunMain();
}
