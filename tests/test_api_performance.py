#!interpreter [optional-arg]
# -*- coding: utf-8 -*-

"""
This contains to load test of Hello World example of gRPC call with locust.
"""

# Built-in/Generic Imports
import sys, os
import grpc
import inspect
import time
import gevent
import numpy as np

# Libs
from locust.contrib.fasthttp import FastHttpUser
from locust import task, events, constant
from locust.runners import STATE_STOPPING, STATE_STOPPED, STATE_CLEANUP, WorkerRunner

sys.path.append(os.path.abspath(".."))
from action_recog_demo.grpc_test import inference_pb2_grpc
from action_recog_demo.grpc_test.torchserve_grpc_client import infer_image


def stopwatch(func):
    """To be updated"""

    def wrapper(*args, **kwargs):
        """To be updated"""
        # get task's function name
        previous_frame = inspect.currentframe().f_back
        _, _, task_name, _, _ = inspect.getframeinfo(previous_frame)

        start = time.time()
        result = None
        try:
            result = func(*args, **kwargs)
        except Exception as e:
            total = int((time.time() - start) * 1000)
            events.request_failure.fire(request_type="TYPE",
                                        name=task_name,
                                        response_time=total,
                                        response_length=0,
                                        exception=e)
        else:
            total = int((time.time() - start) * 1000)
            events.request_success.fire(request_type="TYPE",
                                        name=task_name,
                                        response_time=total,
                                        response_length=0)
        return result

    return wrapper


class GRPCMyLocust(FastHttpUser):
    host = 'http://127.0.0.1:7070'
    wait_time = constant(0)

    def on_start(self):
        """ on_start is called when a Locust start before any task is scheduled """
        pass

    def on_stop(self):
        """ on_stop is called when the TaskSet is stopping """
        pass

    @task
    @stopwatch
    def grpc_client_task(self):
        """To be updated"""
        try:
            with grpc.insecure_channel("127.0.0.1:7070") as channel:
                stub = inference_pb2_grpc.InferenceAPIsServiceStub(channel)

                MODEL_NAME = "STCNet_8f_CesleaFDD6"  # use 8 frames as input
                WIDTH, HEIGHT = 640, 480
                rgb_input = np.random.randint(0, 255, size=(HEIGHT, WIDTH, 3), dtype=np.uint8)

                # Action recognition API Call
                response = infer_image(
                    # gRPC 통신을 위한 부분. ACTION_API: 추론 API 서버의 URI
                    stub,
                    # 추론에 사용할 모델명 (어떤 모델을 사용할 것인가?)
                    MODEL_NAME,
                    # RGB 비디오 프레임 (numpy array)
                    rgb_input)
                print(response)

        except (KeyboardInterrupt, SystemExit):
            sys.exit(0)


# Stopping the locust if a threshold (in this case the fail ratio) is exceeded
def checker(environment):
    while not environment.runner.state in [STATE_STOPPING, STATE_STOPPED, STATE_CLEANUP]:
        time.sleep(1)
        if environment.runner.stats.total.fail_ratio > 0.2:
            print(f"fail ratio was {environment.runner.stats.total.fail_ratio}, quitting")
            environment.runner.quit()
            return


@events.init.add_listener
def on_locust_init(environment, **_kwargs):
    if not isinstance(environment.runner, WorkerRunner):
        gevent.spawn(checker, environment)
