"""Template for ``ship_config.py`` — copy this, do not edit it in place.

    cp ship_config.sample.py ship_config.py

``ship_to_host.py`` reads ``ship_config.py`` from beside itself and keeps
nothing site-specific of its own, so the script can be copied between projects
or published while the credentials stay behind in this one file.

Every entry is optional. Anything left empty falls back to the environment
variable named next to it, and a command-line flag overrides both.

``ship_config.py`` is listed in .gitignore. Keep it that way — it holds a real
password, and this sample is the only one of the pair that belongs in git.
"""

# --- target host -----------------------------------------------------------
HOST       = ""                     # $SHIP_HOST — e.g. "build01.example.net"
USER       = ""                     # $SHIP_USER — remote account
PASSWORD   = ""                     # $SHIP_PASSWORD — empty means key auth
SSH_PORT   = "22"                   # $SHIP_SSH_PORT

# The project root on the host — the counterpart of your checkout.
REMOTE_DIR = "/tmp/{user}/project"  # $SHIP_REMOTE_DIR — {user} comes from USER

# The part of the project to actually send, relative to REMOTE_DIR. Each
# entry is wiped on the host before the new copy lands, so files deleted
# locally disappear there too. Empty sends the whole project, extracting
# over the existing tree rather than replacing it.
#
#     UPLOAD_DIR = "app/rag"              # just that package
#     UPLOAD_DIR = ["app/rag", "agent"]   # several
#     UPLOAD_DIR = ""                     # everything
UPLOAD_DIR = ""                     # $SHIP_UPLOAD_DIR

# --- package index, used only by --setup -----------------------------------
# On JFrog Artifactory the token is the reference token its "Set Me Up" dialog
# hands you for the pypi repo.
ARTIFACTORY_USER  = ""   # $ARTIFACTORY_USER  — index account
ARTIFACTORY_HOST  = ""   # $ARTIFACTORY_HOST  — e.g. "artifacts.example.net"
ARTIFACTORY_TOKEN = ""   # $ARTIFACTORY_TOKEN — secret
