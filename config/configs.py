

def get(name):
    print(name)
    if name == 'setting1':
        from config import setting1 as setting
    elif name == 'setting2':
        from config import setting2 as setting
    elif name == 'setting3':
        from config import setting3 as setting
    elif name == 'setting4':
        from config import setting4 as setting
    elif name == 'setting5':
        from config import setting5 as setting
    elif name == 'setting6':
        from config import setting6 as setting


    cfg = setting.Config()
    return cfg
