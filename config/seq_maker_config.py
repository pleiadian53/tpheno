from os import getenv 

Server = getenv("PYMSSQL_ELIXR_SERVER")
User = getenv("PYMSSQL_USERNAME")
Password = getenv("PYMSSQL_PASSWORD")

if not Server or not User: 
    from config import sys_config
    Server = sys_config.read('PYMSSQL_ELIXR_SERVER', None)
    User = sys_config.read("PYMSSQL_USERNAME", None)
    if not server or not user: 
    	raise ValueError, "No server and user info given!"

if not Password: 
    Password = input('password> ')

def test(): 
    print('current settings: server=%s, user=%s, passwd=%s' % (Server, User, Password))

if __name__ == "__main__": 
    test()

