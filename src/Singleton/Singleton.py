"""Definition of meta model 'Singleton'."""
import sqlite3

from functools import partial
import pyecore.ecore as Ecore
from pyecore.ecore import *


name = 'Singleton'
nsURI = ''
nsPrefix = ''

eClass = EPackage(name=name, nsURI=nsURI, nsPrefix=nsPrefix)

eClassifiers = {}
getEClassifier = partial(Ecore.getEClassifier, searchspace=eClassifiers)


class DatabaseConnection(EObject, metaclass=MetaEClass):
    class __DatabaseConnection:
        def __init__(self):
            self.conn = sqlite3.connect('ocr.db')
            self.conn.isolation_level = None
            try:
                self.conn.execute('''CREATE TABLE users (name text primary key, passwd text)''')
                self.conn.execute('''CREATE TABLE pics (name text, user_name text)''')
            except sqlite3.OperationalError:
                pass
            self.cursor = self.conn.cursor()

        def __del__(self):
            self.cursor.close()

    instance = None

    def __init__(self):
        if not DatabaseConnection.instance:
            DatabaseConnection.instance = DatabaseConnection.__DatabaseConnection()

    def __getattr__(self, name):
        return getattr(self.instance, name)

    def add_user(self, name, passwd):
        self.cursor.execute('''INSERT INTO users (name, passwd) values (?, ?)''', (name, passwd))

    def add_pic(self, user, name):
        raise NotImplementedError('operation add_pic(...) not yet implemented')

    def get_user_pics(self, user_name):
        raise NotImplementedError('operation get_user_pics(...) not yet implemented')

    def delete_user(self, name):
        raise NotImplementedError('operation delete_user(...) not yet implemented')

    def delete_pic(self, name):
        raise NotImplementedError('operation delete_pic(...) not yet implemented')

    def update_password(self, name, new_pass):
        raise NotImplementedError('operation update_password(...) not yet implemented')


def main():
    db = DatabaseConnection()
    db.add_user("Admin", "Admin123")


if __name__ == '__main__':
    main()
