# Prevent custom nodes from hooking anything important
HOOK_BREAK = []


SAVED_FUNCTIONS = []


def save_functions():
    for f in HOOK_BREAK:
        SAVED_FUNCTIONS.append((f[0], f[1], getattr(f[0], f[1])))


def restore_functions():
    for f in SAVED_FUNCTIONS:
        setattr(f[0], f[1], f[2])
