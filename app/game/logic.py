# app/game/logic.py

class GameLogic:
    def __init__(self, command_queue):
        self.command_queue = command_queue
        self.current_instruction = None

        self.possible_frames = []

    def update(self, detections, frame):
        # Aquí se pueden procesar detecciones y actualizar el estado del juego
        pass

    def process_command(self, comando):
        # Analizar comando, marcar objetos, generar respuestas
        self.current_instruction = f"Procesando: {comando}"

    def has_instruction(self):
        return self.current_instruction is not None

    def get_instruction(self):
        instr = self.current_instruction
        self.current_instruction = None
        return instr
