from doctest import master
from time import sleep
from numpy import place
import xmltodict
import json
import os
import tkinter as tk
import tkinter.ttk
from tkinter import *
from tkinter import ttk
import folderManager as Fm
import time
import pickle
from datetime import date
from multiprocessing import Process, Value, Manager
import socket
#from task_and_acustics_data_collection import Main

class Program:
    def __init__(self):
        self.conversion_counter = 0
        self.running = False
        self.path = os.getcwd()
        self.FolderHandler = Fm.FolderManager(self.path)
        
        self.port = 6000
        self.host = '127.0.0.1' #socket.gethostbyname(socket.gethostname())
        #self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s = socket.socket(type=socket.SOCK_DGRAM)
        #self.s.connect((self.host,self.port))
      

    def StartSystem(self):
        self.running = True
        root.after(1,self.SystemLoop)
        background1.itemconfig(status_light,fill = "green")
        proces_type = error_type_entry_field.get()
        woodnumber = wood_number_entry_field.get()
        date_today = date_entry_field.get()
        self.conversion_counter = int(screwcounter_entry.get())
        conversion_counter_label['text'] = 'Data collected: ' + str(self.conversion_counter)
        self.s.sendto((str(self.conversion_counter)+","+str(woodnumber)+","+str(proces_type)+","+str(date_today)).encode(),(self.host,self.port))
        tab4.ElogView.insert(END,"System Started")

        
        
 

    def StopSystem(self):
        self.running = False
        background1.itemconfig(status_light,fill = "red")
      

    def ResetButton(self):
        self.conversion_counter = 0
        conversion_counter_label['text'] = 'Data collected: ' + str(self.conversion_counter)
        tab4.ElogView.insert(END,"System Stopped")

    def SystemLoop(self):
        folder = wsk3_kxml_detect_folder_entry_field.get()
        name= self.FolderHandler.Wsk3FolderDetect(folder)
        

        if str(name) != "-1":
            self.conversion_counter += 1
            json = self.FolderHandler.ConvertKxmlToJson(name)
            proces_type = error_type_entry_field.get()
            woodnumber = wood_number_entry_field.get()
            date_today = date_entry_field.get()
            
            #print(str(date_today))
            tab4.ElogView.insert(END,"Procesing: "+str(date_today)+str(woodnumber)+str(proces_type)+str(self.conversion_counter) + " from " +name)
            self.FolderHandler.SaveKxmlFile(self.conversion_counter,proces_type, str(date_today)+woodnumber,name)
            self.FolderHandler.SaveJsonFile(self.conversion_counter,proces_type, str(date_today)+woodnumber, json)
            
            conversion_counter_label['text'] = 'Data collected: ' + str(self.conversion_counter)
            screwcounter_entry.delete(0,END)
            screwcounter_entry.insert(0,str(self.conversion_counter))

        



        if self.running == True:
            root.after(1,self.SystemLoop)
    def deletefile(self):
        proces_type = error_type_entry_field_tab3.get()
        woodnumber = wood_number_entry_field_tab3.get()
        screw = delete_screw_number_entry_tab3.get()
        date_today = date_entry_field3.get()
        if self.FolderHandler.DeleteFileJsonAndKxml(str(date_today)+woodnumber,proces_type,screw) == True:
          tab4.ElogView.insert(END,"Deleting: " +str(date_today)+str(woodnumber)+str(proces_type)+str(screw) + ".json")
          tab4.ElogView.insert(END,"Deleting: " +str(date_today)+str(woodnumber)+str(proces_type)+str(screw) + ".csv")
          tab4.ElogView.insert(END,"Deleting: " +str(date_today)+str(woodnumber)+str(proces_type)+str(screw) + ".wav")
          tab4.ElogView.insert(END,"Deleting: " +str(date_today)+str(woodnumber)+str(proces_type)+str(screw) + ".kxml")
        else:
          tab4.ElogView.insert(END,"Could not delete: " +str(date_today)+str(woodnumber)+str(proces_type)+str(screw) + ".json")
          tab4.ElogView.insert(END,"Could not delete: " +str(date_today)+str(woodnumber)+str(proces_type)+str(screw) + ".csv")
          tab4.ElogView.insert(END,"Could not delete: " +str(date_today)+str(woodnumber)+str(proces_type)+str(screw) + ".wav")
          tab4.ElogView.insert(END,"Could not delete: " +str(date_today)+str(woodnumber)+str(proces_type)+str(screw) + ".kxml")
    
    def ChangeLabel(self):
        proces_type_old = error_type_entry_field_changing.get()
        woodnumber = wood_number_entry_field_changing.get()
        screw = screw_number_entry_change.get()
        new_process_type = new_process_type_entry.get()
        date_today = date_entry_field2.get()
        if self.FolderHandler.ChangeLabel(str(date_today)+woodnumber,proces_type_old,new_process_type,screw) == True:
          tab4.ElogView.insert(END,"Changing: " +str(date_today)+str(woodnumber)+str(proces_type_old)+str(screw) + ".json" + "  To: " + str(date_today)+str(woodnumber)+str(new_process_type)+str(screw) + ".json")
          tab4.ElogView.insert(END,"Changing: " +str(date_today)+str(woodnumber)+str(proces_type_old)+str(screw) + ".csv" + "  To: " + str(date_today)+str(woodnumber)+str(new_process_type)+str(screw) + ".csv")
          tab4.ElogView.insert(END,"Changing: " +str(date_today)+str(woodnumber)+str(proces_type_old)+str(screw) + ".wav" + "  To: " + str(date_today)+str(woodnumber)+str(new_process_type)+str(screw) + ".wav")
          tab4.ElogView.insert(END,"Changing: " +str(date_today)+str(woodnumber)+str(proces_type_old)+str(screw) + ".kxml" + "  To: " + str(date_today)+str(woodnumber)+str(new_process_type)+str(screw) + ".kxml")
        else:
          tab4.ElogView.insert(END,"Could not changing: " +str(date_today)+str(woodnumber)+str(proces_type_old)+str(screw) + ".json")
          tab4.ElogView.insert(END,"Could not changing: " +str(date_today)+str(woodnumber)+str(proces_type_old)+str(screw) + ".csv")
          tab4.ElogView.insert(END,"Could not changing: " +str(date_today)+str(woodnumber)+str(proces_type_old)+str(screw) + ".wav")
          tab4.ElogView.insert(END,"Could not changing: " +str(date_today)+str(woodnumber)+str(proces_type_old)+str(screw) + ".kxml")
    

##################    tkinter setup   ###################
if __name__ == '__main__':
    
    program = Program()

    root=tk.Tk()
    root.title("Screwing Cell data collector 0.1")
    displayX = root.winfo_screenwidth()
    displayY = root.winfo_screenheight()
    root.resizable(width=True, height=True)
    root.geometry(str(displayX-100)+'x'+str(displayY-100))
    root.winfo_geometry

    tabControl = ttk.Notebook(root)

    tab1 = ttk.Frame(tabControl)
    tab2 = ttk.Frame(tabControl)
    tab3 = ttk.Frame(tabControl)
    tab4 = ttk.Frame(tabControl)
    tabControl.add(tab1,text= 'Data Acquisition')
    tabControl.add(tab2, text= 'Change Label')
    tabControl.add(tab3,text= 'Deleting Samples')
    tabControl.add(tab4,text= 'Log')
    tabControl.pack(pady=10, expand=True)


    background1 = Canvas(tab1, width= displayX-100, height = displayY-100, bg="#000080")
    background1.pack()

    background2 = Canvas(tab2, width= displayX-100, height = displayY-100, bg="#000080")
    background2.pack()

    background3 = Canvas(tab3, width= displayX-100, height = displayY-100, bg="#000080")
    background3.pack()

    background4 = Canvas(tab4, width= displayX-100, height = displayY-100, bg="#000080")
    background4.pack()

    start = Button(tab1,borderwidth=3, text="Start", font = ("Arial", 12, 'bold'),  height=3, width=10, compound='c', bg='#FC9B0C', fg='white', highlightbackground='green', activebackground= 'green', command=program.StartSystem)
    start.place(x=200,y=200)

    stop = Button(tab1,borderwidth=3, text="Stop", font = ("Arial", 12, 'bold'),  height=3, width=10, compound='c', bg='#FC9B0C', fg='white', highlightbackground='green', activebackground= 'green', command=program.StopSystem)
    stop.place(x=200,y=300)

    error_type_label = Label(tab1,text="Process Type:",font = ("Arial", 12, 'bold'))
    error_type_label.place(x=350,y=200)



    error_type_entry_field = Entry(tab1)
    error_type_entry_field.place(x=500,y=200)

    wood_number_label = Label(tab1, text="Wood Number:",font = ("Arial", 12, 'bold'))
    wood_number_label.place(x=370,y=300)

    wood_number_entry_field = Entry(tab1)
    wood_number_entry_field.place(x=500,y=300)

    date_label = Label(tab1, text="Date:", font = ("Arial", 12, 'bold'))
    date_label.place(x=370,y=350)
    date_entry_field = Entry(tab1)
    date_entry_field.place(x=500,y=350)
    date_entry_field.delete(0,END)
    #date_today = date.today()
    #date_entry_field.insert(0,str(date_today.day)+str(date_today.month)+str(date_today.year))
    date_entry_field.insert(0,date.today().strftime('%d%m%Y'))

    
    


    date_label3 = Label(tab3, text="Date:", font = ("Arial", 12, 'bold'))
    date_label3.place(x=370,y=350)
    date_entry_field3 = Entry(tab3)
    date_entry_field3.place(x=500,y=350)
    date_entry_field3.delete(0,END)
    
    #date_entry_field3.insert(0,str(date_today.day)+str(date_today.month)+str(date_today.year))
    date_entry_field3.insert(0,date.today().strftime('%d%m%Y'))

    status_light = background1.create_oval(50, 50, 100, 100)
    background1.itemconfig(status_light,fill = "gray")

    wsk3_kxml_detect_folder_entry_field = Entry(tab1)
    wsk3_kxml_detect_folder_entry_field.place(x=500,y=400)
    wsk3_kxml_detect_folder_entry_field.delete(0,END)
    wsk3_kxml_detect_folder_entry_field.insert(0,"WSK3")

    wsk3_kxml_detect_folder_label = Label(tab1, text="Detecting folder", font= ("Arial", 12, 'bold'))
    wsk3_kxml_detect_folder_label.place(x=370, y=400)


    scroll = Scrollbar()
    scroll.pack(side=RIGHT, fill=Y)
    configfile = Text(tab1, wrap=WORD, width=40, height=40, yscrollcommand=scroll.set)
    filename='information.txt'
    with open(filename, 'r', encoding='utf-8-sig') as file:
      configfile.insert(INSERT, file.read())
    configfile.pack(side="left")
    scroll.config(command=configfile.yview)
    configfile.place(x=displayX-440,y=40)

    conversion_counter_label = Label(tab1,text="Data collected: 0",font = ("Arial", 20, 'bold'))
    conversion_counter_label.place(x=280,y=150)

    start_pinnumber_label = Label(tab1,text="Start pinnumber",font = ("Arial", 12, 'bold'))
    start_pinnumber_label.place(x=550,y=120)

    screwcounter_entry = Entry(tab1,text="0",font = ("Arial", 20, 'bold'))
    screwcounter_entry.place(x=550,y=150)
    screwcounter_entry.delete(0,END)
    screwcounter_entry.insert(0,"0")

    counter_reset_button = Button(tab1,borderwidth=3, text="Reset counter", font = ("Arial", 12, 'bold'),  compound='c', bg='#FC9B0C', fg='white', highlightbackground='green', activebackground= 'green', command=program.ResetButton)
    counter_reset_button.place(x=100,y=150)

    ################### changing label tab ######################

    change = Button(tab2,borderwidth=3, text="Change Label", font = ("Arial", 12, 'bold'),  height=3, width=10, compound='c', bg='#FC9B0C', fg='white', highlightbackground='green', activebackground= 'green', command=program.ChangeLabel)
    change.place(x=200,y=200)

    date_label2 = Label(tab2, text="Date:", font = ("Arial", 12, 'bold'))
    date_label2.place(x=370,y=350)
    date_entry_field2 = Entry(tab2)
    date_entry_field2.place(x=500,y=350)
    date_entry_field2.delete(0,END)
    #date_entry_field2.insert(0,str(date_today.day)+str(date_today.month)+str(date_today.year))
    date_entry_field2.insert(0,date.today().strftime('%d%m%Y'))

    error_type_label_chaning = Label(tab2,text="Old Proces Type:",font = ("Arial", 12, 'bold'))
    error_type_label_chaning.place(x=400,y=200)



    error_type_entry_field_changing = Entry(tab2)
    error_type_entry_field_changing.place(x=550,y=200)

    new_process_type_label = Label(tab2,text="New Proces Type:",font = ("Arial", 12, 'bold'))
    new_process_type_label.place(x=400,y=250)

    new_process_type_entry = Entry(tab2)
    new_process_type_entry.place(x=550 ,y=250)

    wood_number_label_changing = Label(tab2, text="Wood Number:",font = ("Arial", 12, 'bold'))
    wood_number_label_changing.place(x=370,y=300)

    wood_number_entry_field_changing = Entry(tab2)
    wood_number_entry_field_changing.place(x=500,y=300)


    screw_number_entry_change = Entry(tab2)
    screw_number_entry_change.place(x=500,y=400)

    screwnumber_label_change = Label(tab2, text="Screwnumber:", font= ("Arial", 12, 'bold'))
    screwnumber_label_change.place(x=370, y=400)

    


    ######################nchanging label tab ###################
    ############### tab3 #########################

    delete = Button(tab3,borderwidth=3, text="Delete File", font = ("Arial", 12, 'bold'),  height=3, width=10, compound='c', bg='RED', fg='white', highlightbackground='green', activebackground= 'green', command=program.deletefile)
    delete.place(x=200,y=200)

    error_type_label_tab3 = Label(tab3,text="Error Type:",font = ("Arial", 12, 'bold'))
    error_type_label_tab3.place(x=400,y=200)

    error_type_entry_field_tab3 = Entry(tab3)
    error_type_entry_field_tab3.place(x=500,y=200)

    wood_number_label_tab3 = Label(tab3, text="Wood Number:",font = ("Arial", 12, 'bold'))
    wood_number_label_tab3.place(x=370,y=300)

    wood_number_entry_field_tab3 = Entry(tab3)
    wood_number_entry_field_tab3.place(x=500,y=300)


    delete_screw_number_entry_tab3 = Entry(tab3)
    delete_screw_number_entry_tab3.place(x=500,y=400)

    delete_screwnumber_label_tab3 = Label(tab3, text="Screwnumber:", font= ("Arial", 12, 'bold'))
    delete_screwnumber_label_tab3.place(x=370, y=400)


    ############## tab4 ###############

    Elogframe=Frame(tab4)
    Elogframe.place(x=40,y=15)
    scrollbar = Scrollbar(Elogframe) 
    scrollbar.pack(side=RIGHT, fill=Y)
    tab4.ElogView = Listbox(Elogframe,width=150,height=40, yscrollcommand=scrollbar.set)
    try:
      Elog = pickle.load(open("ErrorLog","rb"))
    except:
      Elog = ['##LOG##']
      pickle.dump(Elog,open("ErrorLog","wb"))
    time.sleep(.2)
    for item in Elog:
      tab4.ElogView.insert(END,item) 
    tab4.ElogView.pack()
    scrollbar.config(command=tab4.ElogView.yview)

    def clearLog():
      tab4.ElogView.delete(1,END)
      value=tab4.ElogView.get(0,END)
      pickle.dump(value,open("ErrorLog","wb"))

      clearLogBut = Button(tab4,  text="Clear The Log",  width=26, command = clearLog)
      clearLogBut.place(x=1000, y=630)


    tab4.ElogView.insert(END,"System Ready")

    ##################################

    ############### tkinter setup end ################


    root.mainloop()