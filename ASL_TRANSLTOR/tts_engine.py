import pyttsx3, threading # type: ignore
class TTSEngine:
    def __init__(self):
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)
        self._lock = threading.Lock()

    def speak(self, text):
        def _run():
            with self._lock:
                self.engine.say(text)
                self.engine.runAndWait()
        threading.Thread(target=_run, daemon=True).start()