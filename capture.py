import cv2

class MyVideoCapture:
    def __init__(self, capture_source = 0):
        
        # Open the capture source
        self.cap = cv2.VideoCapture(capture_source)
        if not self.cap.isOpened():
            raise ValueError("Unable to open capture source", capture_source)

        # Get capture source details
        self.width  = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.fps    = self.cap.get(cv2.CAP_PROP_FPS)

    def get_frame(self):
        if self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                # Return a boolean success flag and the current frame
                # return (ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                return (ret, frame)
            else:
                return (ret, None)
        else:
            return (ret, None)

    def __del__(self):
        if self.cap.isOpened():
            # Release the capture source when the object is destroyed
            self.cap.release()