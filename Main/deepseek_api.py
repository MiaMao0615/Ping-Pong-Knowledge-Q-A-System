# -*- coding: utf-8 -*-
"""
API utilities for the ping pong agent.

This module wraps calls to the Tencent Cloud DeepSeek LLM service. It provides
functions for obtaining both streaming and non‑streaming completions given a
conversation history. The agent uses these functions to generate responses
about table tennis rules and related information. All comments and docstrings
in this file are written in English for clarity.
"""
import json
import requests
import time

from tencentcloud.common.common_client import CommonClient
from tencentcloud.common import credential
from tencentcloud.common.exception.tencent_cloud_sdk_exception import TencentCloudSDKException
from tencentcloud.common.profile.client_profile import ClientProfile
from tencentcloud.common.profile.http_profile import HttpProfile

import pdb


class NonStreamResponse(object):
    def __init__(self):
        self.response = ""

    def _deserialize(self, obj):
        self.response = json.dumps(obj)

def get_api_result(messages, secret_id, secret_key, system=None, model='deepseek-v3', service='ap-guangzhou', stream=True):
    try:
        # Instantiate a credential object; you need to pass in your Tencent Cloud account SecretId and SecretKey.
        # Keep your SecretId and SecretKey confidential. Code leakage may expose them and threaten all resources under your account.
        # For secure usage, consult https://cloud.tencent.com/document/product/1278/85305 . Keys can be obtained at https://console.cloud.tencent.com/cam/capi
        cred = credential.Credential(secret_id, secret_key)

        httpProfile = HttpProfile()
        httpProfile.endpoint = "lkeap.tencentcloudapi.com"
        httpProfile.reqTimeout = 40000  # Streaming interfaces may take longer
        clientProfile = ClientProfile()
        clientProfile.httpProfile = httpProfile

        #if system is None:
        #    messages = [{'Role': 'user', 'Content': content}]
        #else:
        #    messages = [{'Role': 'system', 'Content': system}, {'Role': 'user', 'Content': content}]
        params = json.dumps({'Model': model,'Messages': messages, 'Stream': True}, ensure_ascii=False)
        common_client = CommonClient("lkeap", "2024-05-22", cred, service, profile=clientProfile)
        resp = common_client._call_and_deserialize("ChatCompletions", json.loads(params), NonStreamResponse)
        if isinstance(resp, NonStreamResponse):  # Non-streaming response
            return resp.response
        else:  # Streaming response
            for event in resp:
                try:
                    yield json.loads(event['data'])['Choices'][0]['Delta']['Content']#event['data']#event
                except:
                    yield ''
            
    except TencentCloudSDKException as err:
        print(err)

def capture_and_yield(data, secret_id, secret_key, start_marker, end_marker):
    capture, buffer = False, ""
    # Need a flag variable
    for result in get_api_result(data, secret_id, secret_key):
        buffer = buffer + result
        while start_marker in buffer or (end_marker in buffer and capture):
            if start_marker in buffer:
                buffer = ''.join(buffer.split(start_marker, 1)[1:])
                capture = True
            if end_marker in buffer:
                out = buffer.split(end_marker, 1)[0].strip()
                yield out
                buffer = buffer[len(out):]
                capture = False
