import time


def execution_time(function):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        function(*args, **kwargs)
        end_time = time.time()

        print(f"Model Training Time: {end_time - start_time}")

    return wrapper
