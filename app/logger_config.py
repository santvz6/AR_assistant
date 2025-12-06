import logging

logger = logging.getLogger("app_logger")  
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Handler para consola
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# Handler para archivo
file_handler = logging.FileHandler("app/app.log", mode="w")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
