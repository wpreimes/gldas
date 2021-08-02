from gldas.connections import GldasRemote

connection = GldasRemote()
str(connection)
connection.connect('GLDAS_NOAH025_M.2.0')

connection.datedown()