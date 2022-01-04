import cv2

kepala = cv2.CascadeClassifier('haar_frontal_detect.xml')
mulut = cv2.CascadeClassifier('haar_mouth.xml')
video_capture = cv2.VideoCapture(0)

while True:

       # Capture frame-by-frame
   retval, frame = video_capture.read()

   # Convert to grayscale
   gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

   # Detect features specified in Haar Cascade
   deteksi_kepala = kepala.detectMultiScale(
   gray,
   scaleFactor=1.3,
   minNeighbors=5,
   minSize=(20, 20)
   )

   # Draw a rectangle around recognized faces 
   total_kepala = 0
   for (x, y, w, h) in deteksi_kepala:
      cv2.rectangle(frame, (x, y), (x+w, y+h), (200, 50, 50), 2)
      total_kepala = total_kepala + 1
      cv2.putText(frame, ('%02d_kepala Terdeteksi' % total_kepala), (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)
   
      # Detect features specified in Haar Cascade
      roi_gray = gray[y:y+h, x:x+w]
      roi_color = frame[y:y+h, x:x+w]
      deteksi_mulut = mulut.detectMultiScale(roi_gray)
      for (mx,my,mw,mh) in deteksi_mulut:
         cv2.rectangle(roi_color,(mx,my),(mx+mw,my+mh),(0,255,0),2)

      # Display the resulting frame
   cv2.imshow('Video', frame)

   if cv2.waitKey(1) & 0xFF == ord('q'):
         break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()