import os
import xmltodict
import json
import shutil
import pathlib
import time
import shutil

class FolderManager:
    def __init__(self, path):
        self.ext = '.KXML'
        self.path_of_the_directory = path
        print(self.path_of_the_directory)
        self.default_detect_folder = 'WSK3'
        
        
        

    def SaveJsonFile(self, screwnumber, processtype, woodnumber, temp_json):
        if self.CheckIfFolderExist(self.path_of_the_directory + '\\data\\' + str(woodnumber)) == True: #+ '\\' + str(processtype)) == True:
            try:
                with open( "data/" + str(woodnumber) + "/" + str(woodnumber)+str(processtype)+str(screwnumber) + ".json", "w") as json_file:
                    json_file.write(temp_json)
                    json_file.close()
            except OSError:
                print("Error")
        else:
            self.MakeNewFolder('\\data\\' + str(woodnumber))
            try:
                with open( "data/" + str(woodnumber) + "/" + str(woodnumber)+str(processtype)+str(screwnumber) + ".json", "w") as json_file:
                    json_file.write(temp_json)
                    json_file.close()
            except OSError:
                print("Error")

        if not self.CheckIfFolderExist("dashboard"):
                self.MakeNewFolder("dashboard")
                
        # Saving the screwdriver data for the dashboard 
        with open( "dashboard/"+str(woodnumber)+str(processtype)+str(screwnumber) + ".json", "w") as json_file:
                    json_file.write(temp_json)
                    json_file.close()

    def CheckIfFolderExist(self, foldername):
        isdir = os.path.isdir(foldername)
        return isdir


    def MakeNewFolder(self, foldername):       
        path = self.path_of_the_directory+foldername
        try:
            os.makedirs(path)
        except OSError:
            print("Creation of the directory %s failed" % path)
        else:
            print ("Successfully created the directory %s " % path)


    def SaveKxmlFile(self, screwnumber, processtype, woodnumber,name):
        if self.CheckIfFolderExist(self.path_of_the_directory + '\\data_kxml\\' + str(woodnumber)) == True:
            source = self.path_of_the_directory + '\\' + name
            dest = self.path_of_the_directory + '\\data_kxml\\' + str(woodnumber) + '\\' + str(woodnumber)+str(processtype)+str(screwnumber) + '.kxml'
            try:
                os.replace(source, dest)
                #print("Source path renamed to destination path successfully.")
                
            # If Source is a file
            # but destination is a directory
            except IsADirectoryError:
                print("Source is a file but destination is a directory.")
            
            # If source is a directory
            # but destination is a file
            except NotADirectoryError:
                print("Source is a directory but destination is a file.")
            
            # For permission related errors
            except PermissionError:
                print("Operation not permitted.")
            
            # For other errors
            except OSError as error:
                print("OS ERROR")

            self.DeleteFile(name)
        else:
            self.MakeNewFolder('\\data_kxml\\' + str(woodnumber))

            source = self.path_of_the_directory + '\\' + name
            dest = self.path_of_the_directory + '\\data_kxml\\' + str(woodnumber) + '\\' + str(woodnumber)+str(processtype)+str(screwnumber) + '.kxml'
            try:
                os.replace(source, dest)
                #print("Source path renamed to destination path successfully.")
 
            # If Source is a file
            # but destination is a directory
            except IsADirectoryError:
                print("Source is a file but destination is a directory.")
            
            # If source is a directory
            # but destination is a file
            except NotADirectoryError:
                print("Source is a directory but destination is a file.")
            
            # For permission related errors
            except PermissionError:
                print("Operation not permitted.")
            
            # For other errors
            except OSError as error:
                print("OS ERROR")

            self.DeleteFile(name)

    def Wsk3FolderDetect(self, foldername):
        try:
            for filename in os.listdir(self.path_of_the_directory + '\\' + foldername):
                if filename.endswith(self.ext):
                    name = os.path.join(foldername,filename)
                    return name
        except:
            for filename in os.listdir(self.path_of_the_directory + '\\' + self.default_detect_folder):
                if filename.endswith(self.ext):
                    name = os.path.join(self.path_of_the_directory + self.default_detect_folder,filename)
                    return name
        
        
        return -1 # -1 means that there is no kxml in the the folder


    def ConvertKxmlToJson(self, name):
        time.sleep(0.1) # maybe the delay is to high
        try:
            print(name)
            with open(name.replace('\\','/')) as xml_file:
                data_dict = xmltodict.parse(xml_file.read())
                xml_file.close()
                return json.dumps(data_dict)
        except:
            print("could not convert")


    def DeleteFile(self, file_path):
        #filename = os.path.basename(file_path)
        try:
            os.remove(file_path)
            #print("File have been removed successfully")
        except OSError as error:
            print(error)
            print("File path can not be removed")

    def DeleteFileJsonAndKxml(self,woodnumber, proces, screw):
        json_path = 'data\\' + str(woodnumber) +  '\\' + str(woodnumber) + str(proces) + str(screw) + '.json'
        task_path = 'data\\' + str(woodnumber) +  '\\' + str(woodnumber) + str(proces) + str(screw) + '.csv'
        sound_path = 'data\\' + str(woodnumber) +  '\\' + str(woodnumber) + str(proces) + str(screw) + '.wav'
        kxml_path = 'data_kxml\\' + str(woodnumber) + '\\' + str(woodnumber) + str(proces) + str(screw) + '.kxml'
        
        try:
            os.remove(json_path)
            os.remove(task_path)
            os.remove(sound_path)
            os.remove(kxml_path)
            
            #print("File have been removed successfully")
            return True
        except OSError as error:
            print(error)
            print("File path can not be removed")
            return False

    def ChangeLabel(self,woodnumber,proces,new_proces,screw):
        #json = self.ConvertKxmlToJson("data_kxml\\"+str(woodnumber)+"\\"+str(proces)+"\\"+str(woodnumber)+str(proces)+str(screw)+".KXML")
        
        # # reading and writing the json file with the new proces type
        # with open("data\\"+str(woodnumber)+"\\"+str(woodnumber)+str(proces)+str(screw)+".json", "r") as old_file, open("data\\"+str(woodnumber)+"\\"+str(woodnumber)+str(new_proces)+str(screw)+".json", "w") as new_file:
        #         new_file.write(old_file.read())

        # # reading and writing the task csv file with the new proces type
        # with open("data\\"+str(woodnumber)+"\\"+str(woodnumber)+str(proces)+str(screw)+".csv", "r") as old_file, open("data\\"+str(woodnumber)+"\\"+str(woodnumber)+str(new_proces)+str(screw)+".csv", "w") as new_file:
        #         new_file.write(old_file.read())

        # # reading and writing the wav file with the new proces type
        # with open("data\\"+str(woodnumber)+"\\"+str(woodnumber)+str(proces)+str(screw)+".wav", "r") as old_file, open("data\\"+str(woodnumber)+"\\"+str(woodnumber)+str(new_proces)+str(screw)+".wav", "w") as new_file:
        #         new_file.write(old_file.read())        
        # shutil.move("data\\"+str(woodnumber)+"\\"+str(woodnumber)+str(proces)+str(screw)+".json","data\\"+str(woodnumber)+"\\"+str(woodnumber)+str(new_proces)+str(screw)+".json")
        # shutil.move("data\\"+str(woodnumber)+"\\"+str(woodnumber)+str(proces)+str(screw)+".csv","data\\"+str(woodnumber)+"\\"+str(woodnumber)+str(new_proces)+str(screw)+".csv")
        # shutil.move("data\\"+str(woodnumber)+"\\"+str(woodnumber)+str(proces)+str(screw)+".wav","data\\"+str(woodnumber)+"\\"+str(woodnumber)+str(new_proces)+str(screw)+".wav")
        #time.sleep(0.3)
        #json = self.ConvertKxmlToJson("data_kxml\\"+str(woodnumber)+"\\"+str(new_proces)+"\\"+str(woodnumber)+str(new_proces)+str(screw)+".KXML")
        
        
        # if json != None:
        #     self.SaveKxmlFile(screw,new_proces,woodnumber,"data_kxml\\"+str(woodnumber)+"\\"+str(proces)+"\\"+str(woodnumber)+str(proces)+str(screw)+".KXML")
        #     self.SaveJsonFile(screw,new_proces,woodnumber,json)
        #     json_path = 'data\\' + str(woodnumber) + '\\' + str(proces) + '\\' + str(woodnumber) + str(proces) + str(screw) + '.json'
        # #kxml_path = 'data_kxml\\' + str(woodnumber) + '\\' + str(proces) + '\\' + str(woodnumber) + str(proces) + str(screw) + '.kxml'
        
        try:
            shutil.move("data\\"+str(woodnumber)+"\\"+str(woodnumber)+str(proces)+str(screw)+".json","data\\"+str(woodnumber)+"\\"+str(woodnumber)+str(new_proces)+str(screw)+".json")
            shutil.move("data\\"+str(woodnumber)+"\\"+str(woodnumber)+str(proces)+str(screw)+".csv","data\\"+str(woodnumber)+"\\"+str(woodnumber)+str(new_proces)+str(screw)+".csv")
            shutil.move("data\\"+str(woodnumber)+"\\"+str(woodnumber)+str(proces)+str(screw)+".wav","data\\"+str(woodnumber)+"\\"+str(woodnumber)+str(new_proces)+str(screw)+".wav")
            shutil.move("data_kxml\\"+str(woodnumber)+"\\"+str(woodnumber)+str(proces)+str(screw)+".kxml","data_kxml\\"+str(woodnumber)+"\\"+str(woodnumber)+str(new_proces)+str(screw)+".kxml")
            
            # os.remove("data\\"+str(woodnumber)+"\\"+str(woodnumber)+str(proces)+str(screw)+".json")
            # os.remove("data\\"+str(woodnumber)+"\\"+str(woodnumber)+str(proces)+str(screw)+".csv")
            # os.remove("data\\"+str(woodnumber)+"\\"+str(woodnumber)+str(proces)+str(screw)+".wav")
            #print("File have been removed successfully")
            return True
        except OSError as error:
            print(error)
            print("File path can not be removed")
            return False
        # else:
        #     return False