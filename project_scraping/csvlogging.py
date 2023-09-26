import csv  
import os.path
import datetime
import uuid

def uniqueID():
    return str(uuid.uuid4()).split('-')[0]

def log_to_csv(filename,uniqueID,data,tag=""):
    try:
        ct = datetime.datetime.now()
        if os.path.isfile(filename):
            with open(filename, 'a') as f:
                writer = csv.writer(f)
                writer.writerow([ct,uniqueID,data,tag])
        else:
            with open(filename, 'w') as f:
                writer = csv.writer(f)
                writer.writerow(["Timestamp","ID","Data","Tags"])
                writer.writerow([ct,uniqueID,data,tag])
    except:
        print(f"Error in logging: {uniqueID}: {data}, Tag:{tag}")

class Log:
    def __init__(self,filename) -> None:
        self.id = self.uniqueID()
        self.filename = filename
        pass

    def uniqueID(self):
        return str(uuid.uuid4()).split('-')[0]

    def log(self,data,tag=""):
        try:
            ct = datetime.datetime.now()
            if os.path.isfile(self.filename):
                with open(self.filename, 'a') as f:
                    writer = csv.writer(f)
                    writer.writerow([ct,self.id,data,tag])
            else:
                with open(self.filename, 'w') as f:
                    writer = csv.writer(f)
                    writer.writerow(["Timestamp","ID","Data","Tags"])
                    writer.writerow([ct,self.id,data,tag])
        except:
            print(f"Error in logging: {self.id}: {data}, Tag:{tag}")

