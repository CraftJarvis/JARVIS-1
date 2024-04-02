# This script is copied from the MineRL project and is used for the purpose of
# preparing the Minecraft Coding Pack (MCP).
#
# Copyright (c) 2020 All Rights Reserved
# Author: William H. Guss, Brandon Houghton

import os
import json
import shutil

import subprocess

def unpack_assets():
    asset_dir = os.path.join(os.path.expanduser('~'), '.gradle', 'caches', 'forge_gradle', 'assets')
    output_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'jarvis', 'stark_tech', 'MCP-Reborn', 'src', 'main', 'resources')
    index = load_asset_index(os.path.join(asset_dir, 'indexes', '1.16.json'))
    unpack_assets_impl(index, asset_dir, output_dir)

def load_asset_index(index_file):
    with open(index_file) as f:
        return json.load(f)

def unpack_assets_impl(index, asset_dir, output_dir):
    for k, v in index['objects'].items():
        asset_hash = v["hash"]
        src = os.path.join(asset_dir, 'objects', asset_hash[:2], asset_hash)
        dst = os.path.join(output_dir, k)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy(src, dst)

def prep_mcp():
    mydir = os.path.abspath(os.path.dirname(__file__))

    # First, get MCP and patch it with our source.
    if os.name == 'nt':
        # Windows is picky about this, too... If you have WSL, you have
        # bash command, but an absolute path won't work. So lets instead
        # use relative paths
        old_dir = os.getcwd()
        os.chdir(os.path.join(mydir, 'scripts'))

        try:
            setup_output = subprocess.check_output(['bash.exe', 'setup_mcp.sh']).decode(errors="ignore")
            if "ERROR: JAVA_HOME" in setup_output:
                raise RuntimeError(
                    """
                    `java` and/or `javac` commands were not found by the installation script.
                    Make sure you have installed Java JDK 8.
                    On Windows, if you installed WSL/WSL2, you may need to install JDK 8 in your WSL
                    environment with `sudo apt update; sudo apt install openjdk-8-jdk`.
                    """
                )
            elif "Cannot lock task history" in setup_output:
                raise RuntimeError(
                    """
                    Installation failed probably due to Java processes dangling around from previous attempts.
                    Try killing all Java processes in Windows and WSL (if you use it). Rebooting machine
                    should also work.
                    """
                )
            subprocess.check_call(['bash.exe', 'patch_mcp.sh'])
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                """
                Running install scripts failed. Check error logs above for more information.

                If errors are about `bash` command not found, You have at least two options to fix this:
                 1. Install Windows Subsystem for Linux (WSL. Tested on WSL 2). Note that installation with WSL
                    may seem especially slow/stuck, but it is not; it is just a bit slow.
                 2. Install bash along some other tools. E.g., git will come with bash: https://git-scm.com/downloads .
                    After installation, you may have to update environment variables to include a path which contains
                    'bash.exe'. For above git tools, this is [installation-dir]/bin.
                After installation, you should have 'bash' command in your command line/powershell.

                If errors are about "could not create work tree dir...", try cloning the MineRL repository
                to a different location and try installation again.
                """
            )

        os.chdir(old_dir)
    else:
        subprocess.check_call(['bash', os.path.join(mydir, 'scripts', 'setup_mcp.sh')])
        subprocess.check_call(['bash', os.path.join(mydir, 'scripts', 'patch_mcp.sh')])

    # Next, move onto building the MCP source
    gradlew = 'gradlew.bat' if os.name == 'nt' else './gradlew'
    workdir = os.path.join(mydir, 'jarvis', 'stark_tech', 'MCP-Reborn')
    if os.name == 'nt':
        # Windows is picky about being in the right directory to run gradle
        old_dir = os.getcwd()
        os.chdir(workdir)
    
    # This may fail on the first try. Try few times
    n_trials = 3
    for i in range(n_trials):
        try:
            subprocess.check_call('{} downloadAssets'.format(gradlew).split(' '), cwd=workdir)
        except subprocess.CalledProcessError as e:
            if i == n_trials - 1:
                raise e
        else:
            break

    unpack_assets()
    subprocess.check_call('{} clean build shadowJar'.format(gradlew).split(' '), cwd=workdir)
    if os.name == 'nt':
        os.chdir(old_dir)

if __name__ == '__main__':
    prep_mcp()