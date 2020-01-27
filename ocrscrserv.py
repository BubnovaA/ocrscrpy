import socket
import json
import base64
import os
import cv2
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import pickle
import requests
import io
from http.server import BaseHTTPRequestHandler, HTTPServer
import logging
import cgi
from pprint import pprint


class S(BaseHTTPRequestHandler):
    def _set_response(self, text):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(text.encode('utf-8'))

    def do_POST(self):
        logging.info("POST request,\nPath: %s\nHeaders:\n%s\n", str(self.path), str(self.headers))
        ctype, pdict = cgi.parse_header(self.headers['content-type'])
        pdict['boundary'] = bytes(pdict['boundary'], "utf-8")

        if ctype == 'multipart/form-data':
            fields = cgi.parse_multipart(self.rfile, pdict)

            ocrspace_token = "e69b2d014b88957"
            ScrToTxt = ScreenToTxt(ocrspace_token)
            res = ScrToTxt(fields['image'][0])

            self._set_response(res.to_csv())

        return True


def run(server_class=HTTPServer, handler_class=S, port=80):
    logging.basicConfig(level=logging.INFO)
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    logging.info('Starting httpd...\n')
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    httpd.server_close()
    logging.info('Stopping httpd...\n')


class ScreenToTxt:
    
    def __init__(self, ocr_token):
        self.ocr_token=ocr_token
        self.url_api = "https://api.ocr.space/parse/image"
        self.img_size=[]
        self.name_from_file=[]
        

    def viewImage(self, image, name_of_window):
        cv2.imshow(name_of_window, image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def segmentImage(self):
        print('segmentImage...')
        img = self.img
        reimg = img.reshape((-1,3))
        reimg = np.float32(reimg)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        num_clusters = 2
        _, label, center = cv2.kmeans(reimg, num_clusters, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        center = np.uint8(center)
        res = center[label.flatten()].reshape((img.shape))
        res_hsv = cv2.cvtColor(res, cv2.COLOR_BGR2HSV)
        blurred = cv2.GaussianBlur(res_hsv, (51, 51), 0)
        grey = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY) 
        _, thresh = cv2.threshold(grey, 0, 255, cv2.THRESH_BINARY_INV +
                            cv2.THRESH_OTSU) 
        contours, _ = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        
        list_countours=[]
        for c in contours:
            x,y,w,h = cv2.boundingRect(c)
            if x and y:
                list_countours.append([x,y,w,h])
            else: self.img_size=[w,h]
        return np.array(list_countours)
        

    def segmentRoi(self):
        #select the area with the game result
        print('segmentRoi...')
        narray=self.segmentImage()
        with open('finalized_model.sav', 'rb', ) as model_file:
            model = pickle.load(model_file)
        df = pd.DataFrame(narray)
        all_predict=model.predict(narray)
        df['k']=all_predict
        df['xw']=df.iloc[:,0]+df.iloc[:,2]
        y=df.iloc[:,1].min()
        h=df.iloc[:,1].max()
        x_delta=5
        z_delta=70
        if df[df['k']==1][0].min()<df[df['k']==0][0].min():
            x=int(df[df['k']==1].xw.max())+x_delta
            w=int(df[df['k']==0][0].min()) - x
            z=int(df[df['k']==0][0].max())+z_delta
        else:
            x=int(df[df['k']==0].xw.max())+x_delta
            w=int(df[df['k']==1][0].min()) - x
            z=int(df[df['k']==1][0].max())+z_delta
         
        roi_player=self.img[y:y+h, x:x+w]
        roi_value=self.img[y:y+h, z:self.img_size[0]]
        roi=np.concatenate((roi_player,roi_value), axis=1)
        #self.viewImage(roi, 'roi') 
        return roi

    def ocr(self, img, lng='eng'):
        print('ocr...')
        height, widht, _ = img.shape
        _, compressedimage = cv2.imencode(".jpg", img, [1, 90])
        file_bytes = io.BytesIO(compressedimage)
        result = requests.post(self.url_api, files={"screenshot.jpg": file_bytes},
                               data={"apikey": self.ocr_token, "language": lng, "isTable": True})
        
        result=result.content.decode()
        result = json.loads(result)
        text_detected = result.get("ParsedResults")[0].get("ParsedText")

        pprint(text_detected)

        txt_lst=[i.split('\t') for i in text_detected.split('\r\n')]
        txt_df=pd.DataFrame(txt_lst[1:])
        txt_df.drop(txt_df.columns[-1], axis=1, inplace=True)
        txt_df=txt_df.dropna()
        txt_df.columns=['Player','Value','Gold','Round','W-L']
        # txt_df['Player']=txt_df['Player'].apply(lambda x: x.replace(' ', '')[:4])
        return txt_df
        
    def findName(self, value_play):
        new_name=[]
        for name in value_play:
             new_name.append(next((s for s in self.name_from_file if name.upper() in s.upper()), None))
        return new_name

    def __call__(self, image_data, txt_file=None):
        print('start...')
        df=pd.DataFrame()
        dfname=None
        try:
            # open file
            jpg_as_np = np.frombuffer(image_data, dtype=np.uint8)
            self.img=cv2.imdecode(jpg_as_np, flags=1)
            # print(image_data)
            # segmentation
            img_roi = self.segmentRoi()
        except Exception as e:
            print(e)
            
        else:
            try:
                #text recognition 
                # df=self.ocr(img_roi, 'eng')
                df = self.ocr(img_roi, 'rus')
                return df

                # df = df_rus
                # value_play=list(df['Player'])
                # value_play_rus=list(df_rus['Player'])
                        
                #if exist txt_file - compare result with a file
                # if txt_file:
                #     with open(txt_file, 'r', encoding='utf-8') as f:
                #         for row in f:
                #             self.name_from_file.extend(row.rstrip().split(' '))
                #     new_name=self.findName(value_play)
                #     new_name_rus=self.findName(value_play_rus)
                #     for index, name in enumerate(new_name):
                #         if not name:
                #             new_name[index]=new_name_rus[index]
                #     df['Player']=new_name
                    
                #if not txt_file -  two columns with names in 'eng' and 'rus'  
                #else:
                #    df['PlayerRus']=df_rus['Player']
                # name_pict=image_file.rsplit('.', 1)
                # dfname='data.csv'
                # df.to_csv(dfname)
                # pprint(df)
            except Exception as e:
                print(e)
            
        return None


class Listener:
    def __init__(self, ip, port):
        run(port=port)
        # listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        # listener.bind(('', port))  # IP
        # listener.listen(0)
        # self.imgfile=None
        # self.txtfile=None
        # print('[+] Waiting for incoming connection...')
        # self.connection, address = listener.accept()
        # print('[+] Got a connection from ' + str(address))

    def reliable_send(self, data: bytes) -> None:
        # Разбиваем передаваемые данные на куски максимальной длины 0xffff (65535)
        for chunk in (data[_:_+0xffff] for _ in range(0, len(data), 0xffff)):
            self.connection.send(len(chunk).to_bytes(2, "big")) # Отправляем длину куска (2 байта)
            self.connection.send(chunk) # Отправляем сам кусок
        self.connection.send(b"\x00\x00") # Обозначаем конец передачи куском нулевой длины

    def readexactly(self, bytes_count: int) -> bytes:
        b = b''
        while len(b) < bytes_count: # Пока не получили нужное количество байт
            part = self.connection.recv(bytes_count - len(b)) # Получаем оставшиеся байты
            if not part: # Если из сокета ничего не пришло, значит его закрыли с другой стороны
                raise IOError("Соединение потеряно")
            b += part
        return b

    def reliable_receive(self) -> bytes:
        b = b''
        while True:
            part_len = int.from_bytes(self.readexactly(2), "big") # Определяем длину ожидаемого куска
            if part_len == 0:
                # Если пришёл кусок нулевой длины, то приём окончен
                return b
            b += self.readexactly(part_len) # Считываем сам кусок

    def execute(self):
        #json_data = json.dumps(command)
        #self.reliable_send(command.encode('utf-8'))
        return self.reliable_receive()

    def write_file(self, path, content):
        with open(path, 'wb') as file:
            file.write(base64.b64decode(content))
            return True

    def read_file(self, path):
        with open(path, 'rb') as file:
            #return base64.encodebytes(file.read()).decode("utf-8")
            return base64.b64encode(file.read())

    def run_ocr(self):
        try:
            ocrspace_token = "e69b2d014b88957"
            ScrToTxt = ScreenToTxt(ocrspace_token)
            res=ScrToTxt(self.imgfile, self.txtfile)
        except:
            return False
        finally:
            return res

    def run(self):
        dirpath = os.path.abspath(os.curdir)
        while True:
            
            result = self.execute()
            file = result.decode()
            if file in ['.jpg','.png','.txt']:
                path=dirpath+'\\data'+'\\file'+file
                result = self.execute()
                result = self.write_file(path, result)
                if result:
                    if file=='.txt':
                        self.txtfile=path
                    elif file in ['.jpg','.png']:
                        self.imgfile=path
                    
            elif result.decode()=='start':
                res=self.run_ocr()
                if res:
                    command_result = self.read_file(dirpath+'\\data'+'\\file.csv')
                    self.reliable_send(command_result)
                    print('result send')
                    self.connection.close()
                    break
                else:
                    print('result false')
                    self.connection.close()
                    break
                    

host = socket.gethostname()
port = 8088

my_listener = Listener('0.0.0.0', port)
my_listener.run()

        
