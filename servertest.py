import socket
import time
import select
import matplotlib.pyplot as plt
import threading



# HOST = "127.0.0.1"  # The server's hostname or IP address
# PORT = 6000 # The port used by the server
# # #s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# s = socket.socket(type=socket.SOCK_DGRAM)
# s.bind((HOST, PORT))
#s.listen()
#conn, addr = s.accept()

class TaskSoundServer(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.daemon = True
        HOST = "127.0.0.1"  # The server's hostname or IP address
        PORT = 6000 # The port used by the server
        #s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s = socket.socket(type=socket.SOCK_DGRAM)
        self.s.bind((HOST, PORT))
        self.flag = False
        self.data = 0
    
    def run(self):
        while True:

            # Establish connection with client.
              
            self.data = self.s.recv(1024).decode()
            #print(self.data)
            if  self.data != 0:
                #print(self.data)
                self.flag = True
                
                


    def reedgui(self):

        
        if self.flag:
            self.flag =False
            return self.data
        else:
            return 0


t_array2 = []
t3 = 0
t1 = 0
counter = 0
sever = TaskSoundServer()
sever.start()
#print(f"Connected by {addr}")
print("thread started")
while True:
    t1 = time.monotonic()
    t_array2.append(t1-t3)
    
    t3 = t1
    data = sever.reedgui()
    if str(data) != "0":
        #print(data)
        data = 0

        
    counter += 1
    if counter == 5000:
        break
    
    #print(".")

   


    #time.sleep(2)
    #print("hejsa")
    # conn.sendall(data)
plt.plot(t_array2[1:],".")
plt.xlabel("samples")
plt.ylabel("seconds getting gui info status")
plt.show()