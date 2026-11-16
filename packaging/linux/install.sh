#!/bin/sh
# UtauCover installer for Linux.
#
#   ./install.sh               install for you: ~/.local/share/UtauCover, menu entry, `utaucover` command
#   sudo ./install.sh          install for everyone: /opt/UtauCover, /usr/local/bin/utaucover
#   ./install.sh --uninstall   remove it again (models, stem cache and settings in ~/.local/share/UtauCover
#                              are kept unless you add --purge)
#   ./install.sh --check       only check that the system libraries the window needs are present
#
# UtauCover itself is self-contained (its own Python, libraries and ffmpeg). The window uses Qt, which needs
# a handful of common desktop libraries (X11/xcb, OpenGL, fonts, NSS, ALSA). Most desktops have them all;
# if some are missing this script names them and offers to install them with your package manager.
set -eu

SRC="$(dirname "$(readlink -f "$0")")"
APP_ID=utaucover
if [ "$(id -u)" = 0 ]; then
    DEST=/opt/UtauCover; BIN=/usr/local/bin; SHARE=/usr/share; SUDO=""
else
    DEST="${XDG_DATA_HOME:-$HOME/.local/share}/UtauCover/app"; BIN="$HOME/.local/bin"
    SHARE="${XDG_DATA_HOME:-$HOME/.local/share}"; SUDO="sudo"
fi
DESKTOP="$SHARE/applications/$APP_ID.desktop"
ICON="$SHARE/icons/hicolor/256x256/apps/$APP_ID.png"

say() { printf '%s\n' "$*"; }

# ---------------------------------------------------------------------------- system library check
# soname -> Debian/Ubuntu package (dnf/zypper install sonames directly; pacman uses its own names)
apt_pkg() {
    case "$1" in
        libxcb-cursor.so.0) echo libxcb-cursor0 ;; libxkbcommon-x11.so.0) echo libxkbcommon-x11-0 ;;
        libxkbcommon.so.0) echo libxkbcommon0 ;; libxcb-icccm.so.4) echo libxcb-icccm4 ;;
        libxcb-image.so.0) echo libxcb-image0 ;; libxcb-keysyms.so.1) echo libxcb-keysyms1 ;;
        libxcb-render-util.so.0) echo libxcb-render-util0 ;; libxcb-util.so.1) echo libxcb-util1 ;;
        libxcb-*.so.*) n="${1#libxcb-}"; echo "libxcb-${n%%.so.*}${1##*.so.}" ;;
        libxcb.so.1) echo libxcb1 ;; libX11-xcb.so.1) echo libx11-xcb1 ;; libX11.so.6) echo libx11-6 ;;
        libXext.so.6) echo libxext6 ;; libXfixes.so.3) echo libxfixes3 ;; libXrender.so.1) echo libxrender1 ;;
        libXi.so.6) echo libxi6 ;; libXcomposite.so.1) echo libxcomposite1 ;; libXdamage.so.1) echo libxdamage1 ;;
        libXrandr.so.2) echo libxrandr2 ;; libXtst.so.6) echo libxtst6 ;; libxkbfile.so.1) echo libxkbfile1 ;;
        libxshmfence.so.1) echo libxshmfence1 ;; libgbm.so.1) echo libgbm1 ;; libdrm.so.2) echo libdrm2 ;;
        libEGL.so.1) echo libegl1 ;; libGL.so.1) echo libgl1 ;; libOpenGL.so.0) echo libopengl0 ;;
        libGLX.so.0) echo libglx0 ;; libfontconfig.so.1) echo libfontconfig1 ;; libfreetype.so.6) echo libfreetype6 ;;
        libdbus-1.so.3) echo libdbus-1-3 ;; libnss3.so|libnssutil3.so|libsmime3.so) echo libnss3 ;;
        libnspr4.so|libplc4.so|libplds4.so) echo libnspr4 ;; libasound.so.2) echo libasound2 ;;
        libglib-2.0.so.0|libgthread-2.0.so.0|libgobject-2.0.so.0|libgio-2.0.so.0) echo libglib2.0-0 ;;
        libwayland-client.so.0) echo libwayland-client0 ;; libwayland-cursor.so.0) echo libwayland-cursor0 ;;
        libwayland-egl.so.1) echo libwayland-egl1 ;; libSM.so.6) echo libsm6 ;; libICE.so.6) echo libice6 ;;
        libexpat.so.1) echo libexpat1 ;; libz.so.1) echo zlib1g ;; libkrb5.so.3|libgssapi_krb5.so.2) echo libgssapi-krb5-2 ;;
        *) echo "" ;;
    esac
}
pacman_pkg() {
    case "$1" in
        libxcb-cursor.so.0) echo xcb-util-cursor ;; libxcb-icccm.so.4) echo xcb-util-wm ;;
        libxcb-image.so.0) echo xcb-util-image ;; libxcb-keysyms.so.1) echo xcb-util-keysyms ;;
        libxcb-render-util.so.0) echo xcb-util-renderutil ;; libxcb-util.so.1) echo xcb-util ;; libxcb*.so.*) echo libxcb ;;
        libxkbcommon*.so.0) echo libxkbcommon-x11 ;; libX11*.so.*) echo libx11 ;; libXext.so.6) echo libxext ;;
        libXfixes.so.3) echo libxfixes ;; libXrender.so.1) echo libxrender ;; libXi.so.6) echo libxi ;;
        libXcomposite.so.1) echo libxcomposite ;; libXdamage.so.1) echo libxdamage ;; libXrandr.so.2) echo libxrandr ;;
        libXtst.so.6) echo libxtst ;; libxkbfile.so.1) echo libxkbfile ;; libxshmfence.so.1) echo libxshmfence ;;
        libgbm.so.1) echo mesa ;; libdrm.so.2) echo libdrm ;; libEGL.so.1|libGL.so.1|libOpenGL.so.0|libGLX.so.0) echo libglvnd ;;
        libfontconfig.so.1) echo fontconfig ;; libfreetype.so.6) echo freetype2 ;; libdbus-1.so.3) echo dbus ;;
        libnss3.so|libnssutil3.so|libsmime3.so) echo nss ;; libnspr4.so|libplc4.so|libplds4.so) echo nspr ;;
        libasound.so.2) echo alsa-lib ;; lib*-2.0.so.0) echo glib2 ;; libwayland*) echo wayland ;;
        libSM.so.6) echo libsm ;; libICE.so.6) echo libice ;; libexpat.so.1) echo expat ;; libz.so.1) echo zlib ;;
        libkrb5.so.3|libgssapi_krb5.so.2) echo krb5 ;;
        *) echo "" ;;
    esac
}

missing_libs() {
    QT="$SRC/runtime/lib/python3.11/site-packages/PySide6/Qt"
    for f in "$QT/plugins/platforms/libqxcb.so" "$QT/lib/libQt6XcbQpa.so.6" "$QT/lib/libQt6WebEngineCore.so.6" \
             "$QT/lib/libQt6Gui.so.6" "$QT/lib/libQt6OpenGL.so.6" "$QT/lib/libQt6DBus.so.6" \
             "$SRC/runtime/lib/python3.11/site-packages/PySide6/QtWebEngineWidgets.abi3.so"; do
        [ -e "$f" ] || continue
        LD_LIBRARY_PATH="$QT/lib" ldd "$f" 2>/dev/null | awk '/not found/ {print $1}'
    done | sort -u
}

check_deps() {
    if ! command -v ldd >/dev/null 2>&1; then say "(ldd not found: skipping the library check)"; return 0; fi
    glibc="$(ldd --version 2>/dev/null | head -n1 | grep -oE '[0-9]+\.[0-9]+$' || true)"
    if [ -n "$glibc" ] && [ "$(printf '%s\n2.28\n' "$glibc" | sort -V | head -n1)" != "2.28" ]; then
        say "WARNING: this system has glibc $glibc; UtauCover needs 2.28 or newer (Ubuntu 20.04+, Debian 10+, Fedora 29+, RHEL 8+)."
    fi
    libs="$(missing_libs)"
    if [ -z "$libs" ]; then say "System libraries: all present."; return 0; fi
    say "The window needs these system libraries, which aren't installed:"
    for l in $libs; do say "  $l"; done
    cmd=""
    if command -v apt-get >/dev/null 2>&1; then
        pkgs=""; for l in $libs; do p="$(apt_pkg "$l")"; [ -n "$p" ] && pkgs="$pkgs $p"; done
        [ -n "$pkgs" ] && cmd="apt-get install -y$(printf '%s\n' $pkgs | sort -u | tr '\n' ' ' | sed 's/^/ /')"
    elif command -v dnf >/dev/null 2>&1; then
        caps=""; for l in $libs; do caps="$caps '$l()(64bit)'"; done; cmd="dnf install -y$caps"
    elif command -v zypper >/dev/null 2>&1; then
        caps=""; for l in $libs; do caps="$caps '$l()(64bit)'"; done; cmd="zypper --non-interactive install$caps"
    elif command -v pacman >/dev/null 2>&1; then
        pkgs=""; for l in $libs; do p="$(pacman_pkg "$l")"; [ -n "$p" ] && pkgs="$pkgs $p"; done
        [ -n "$pkgs" ] && cmd="pacman -S --needed --noconfirm$(printf '%s\n' $pkgs | sort -u | tr '\n' ' ' | sed 's/^/ /')"
    fi
    if [ -z "$cmd" ]; then
        say "Install the packages that provide them with your distribution's package manager, then run UtauCover."
        return 0
    fi
    say ""
    say "To install them:  ${SUDO:+sudo }$cmd"
    if [ -t 0 ]; then
        printf 'Install them now? [Y/n] '
        read -r ans || ans=n
        case "$ans" in
            n*|N*) say "Skipped. The command line still works; the window needs these first." ;;
            *) if command -v apt-get >/dev/null 2>&1; then ${SUDO} apt-get update -qq || true; fi
               eval "${SUDO} $cmd" ;;
        esac
    fi
}

# ------------------------------------------------------------------------------------ (un)install
uninstall() {
    rm -f "$BIN/utaucover" "$DESKTOP" "$ICON"
    rm -rf "$DEST"
    if [ "${1:-}" = "--purge" ] && [ "$(id -u)" != 0 ]; then
        rm -rf "${XDG_DATA_HOME:-$HOME/.local/share}/UtauCover"
        say "Removed UtauCover and its data (models, stem cache, settings)."
    else
        say "Removed UtauCover. Models, stem cache and settings are kept in ${XDG_DATA_HOME:-$HOME/.local/share}/UtauCover (./install.sh --uninstall --purge removes them)."
    fi
    command -v update-desktop-database >/dev/null 2>&1 && update-desktop-database -q "$SHARE/applications" 2>/dev/null || true
}

install_app() {
    say "Installing UtauCover to $DEST"
    if [ "$SRC" != "$DEST" ]; then
        rm -rf "$DEST.new"
        mkdir -p "$(dirname "$DEST")"
        cp -a "$SRC" "$DEST.new"
        rm -rf "$DEST"
        mv "$DEST.new" "$DEST"
    fi
    mkdir -p "$BIN" "$(dirname "$DESKTOP")" "$(dirname "$ICON")"
    ln -sfn "$DEST/utaucover-cli" "$BIN/utaucover"
    cp "$DEST/resources/utaucover.png" "$ICON"
    cat > "$DESKTOP" <<EOF
[Desktop Entry]
Type=Application
Name=UtauCover
GenericName=Cover project builder
Comment=Split a song into stems and chart its vocals for OpenUtau / UtauV
Exec="$DEST/UtauCover" %F
Icon=$APP_ID
Terminal=false
Categories=AudioVideo;Audio;Music;
MimeType=audio/mpeg;audio/x-wav;audio/wav;audio/flac;audio/ogg;audio/mp4;audio/x-m4a;
StartupWMClass=UtauCover
StartupNotify=true
EOF
    command -v update-desktop-database >/dev/null 2>&1 && update-desktop-database -q "$SHARE/applications" 2>/dev/null || true
    command -v gtk-update-icon-cache >/dev/null 2>&1 && gtk-update-icon-cache -q "$SHARE/icons/hicolor" 2>/dev/null || true
    say "Installed. Open UtauCover from your applications menu, or run: utaucover --help"
    case ":$PATH:" in *":$BIN:"*) ;; *) say "(Add $BIN to your PATH to use the utaucover command.)" ;; esac
}

case "${1:-}" in
    --uninstall) uninstall "${2:-}" ;;
    --check) check_deps ;;
    --no-deps) install_app ;;
    ""|--install) check_deps; install_app ;;
    -h|--help) sed -n '2,12p' "$0" ;;
    *) say "unknown option: $1 (see --help)"; exit 2 ;;
esac
