from ai2thor.controller import Controller

c = Controller()
print('here')
c.start()
print('here')
event = c.step(dict(action="RotateLeft"))
print('stepped')
event = c.step(dict(action="RotateLeft"))
print('stepped')

assert event.frame.shape == (300, 300, 3)
print(event.frame.shape)
print("Everything works!!!")


# import ai2thor.controller
# from pprint import pprint
# c = ai2thor.controller.Controller()
# event = c.start()
# pprint(event.metadata['agent'])
# event = c.step(dict(action='RotateLeft'))
# pprint(event.metadata['agent'])

# SocketException: The socket has been shut down
#   at System.Net.Sockets.Socket.Send (System.Byte[] buf) [0x00000] in <filename >
#   at AgentManager+<EmitFrame>c__Iterator3.MoveNext () [0x00000] in <filename un>
#   at UnityEngine.SetupCoroutine.InvokeMoveNext (IEnumerator enumerator, IntPtr >

# (Filename:  Line: -1)
