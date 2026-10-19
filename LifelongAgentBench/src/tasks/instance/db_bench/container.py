import docker
import mysql.connector
import random
import socket
import time
from docker.models import containers
from typing import Optional


class DBBenchContainer:
    port = 13000
    password = "password"

    def __init__(self, image: str = "mysql:8.0"):
        self.deleted = False
        self.image = image
        self.client = docker.from_env()
        p = DBBenchContainer.port + random.randint(0, 10000)
        while self.is_port_open(p):
            p += random.randint(0, 20)
        self.port = p
        self.container: containers.Container = self.client.containers.run(
            image,
            command="--innodb-use-native-aio=0",
            name=f"mysql_{self.port}",
            environment={"MYSQL_ROOT_PASSWORD": self.password},
            ports={"3306": self.port},
            detach=True,
            tty=True,
            stdin_open=True,
            remove=True,
        )

        time.sleep(1)

        retry = 0
        while True:
            try:
                self.container.reload()
                if self.container.status == "exited":
                    logs = self.container.logs(tail=80).decode(errors="replace")
                    raise RuntimeError(f"MySQL container exited during startup:\n{logs}")
                self.conn = mysql.connector.connect(
                    host="127.0.0.1",
                    user="root",
                    password=self.password,
                    port=self.port,
                    pool_reset_session=True,
                )
            except (
                mysql.connector.errors.DatabaseError,
                mysql.connector.errors.OperationalError,
            ):
                if retry > 60:
                    raise
                time.sleep(3)
            except mysql.connector.InterfaceError:
                if retry > 60:
                    raise
                time.sleep(3)
            else:
                break
            retry += 1

    def delete(self) -> None:
        self.container.stop()
        self.deleted = True

    def __del__(self) -> None:
        try:
            if not self.deleted:
                self.delete()
        except Exception:  # noqa
            pass

    def execute(
        self,
        multiple_sql: str,
        database: Optional[str] = None,
    ) -> str:
        try:
            return str(self.execute_fetchall(multiple_sql, database))
        except Exception as e:
            return str(e)

    def execute_fetchall(
        self,
        multiple_sql: str,
        database: Optional[str] = None,
    ) -> list[tuple]:
        def fetchall_or_empty() -> list[tuple]:
            try:
                return cursor.fetchall()
            except mysql.connector.InterfaceError as error:
                if "No result set to fetch" not in str(error):
                    raise
                return []

        self.conn.reconnect()
        cursor = self.conn.cursor()
        if database:
            cursor.execute(f"use `{database}`;")
            fetchall_or_empty()
        sql_list = multiple_sql.split(";")
        sql_list = [sql.strip() for sql in sql_list if sql.strip() != ""]
        result: list[tuple] = []
        for sql in sql_list:
            cursor.execute(sql)
            result = fetchall_or_empty()
            self.conn.commit()
        return result

    def is_port_open(
        self, port: int
    ) -> bool:  # noqa (The quality checker of the IDE is wrong)
        try:
            self.client.containers.get(f"mysql_{port}")
            return True
        except Exception:  # noqa
            pass

        # Create a socket object
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # use IPv4 and TCP
        try:
            # Try to connect to the specified port
            sock.connect(("localhost", port))
            # If the connection succeeds, the port is occupied
            return True
        except ConnectionRefusedError:
            # If the connection is refused, the port is not occupied
            return False
        finally:
            # Close the socket
            sock.close()
