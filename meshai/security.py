from cryptography.fernet import Fernet

def generate_key():
    """
    Generates a new encryption key.
    """
    return Fernet.generate_key()

def save_key(key, key_path='./encryption_key.key'):
    """
    Saves the encryption key to a file.
    """
    with open(key_path, 'wb') as key_file:
        key_file.write(key)

def load_key(key_path='./encryption_key.key'):
    """
    Loads the encryption key from a file.
    """
    with open(key_path, 'rb') as key_file:
        return key_file.read()

def encrypt_file(file_path, key):
    """
    Encrypts a file at the given path using the provided key.
    """
    fernet = Fernet(key)
    with open(file_path, 'rb') as file:
        original = file.read()
    encrypted = fernet.encrypt(original)
    with open(file_path, 'wb') as encrypted_file:
        encrypted_file.write(encrypted)

def decrypt_file(file_path, key):
    """
    Decrypts a file at the given path using the provided key.
    """
    fernet = Fernet(key)
    with open(file_path, 'rb') as enc_file:
        encrypted = enc_file.read()
    decrypted = fernet.decrypt(encrypted)
    with open(file_path, 'wb') as dec_file:
        dec_file.write(decrypted)
