import logging


def log_metrics(epoch, train_loss, val_loss, val_f1):
    logging.info(f'{epoch},{train_loss},{val_loss},{val_f1}')

def configure_logger(log_file):
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(message)s')