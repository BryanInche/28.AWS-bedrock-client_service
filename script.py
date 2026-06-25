import pandas as pd
import boto3
import streamlit as st
import json
import requests
import os
import base64

region = "us-east-1"
os.environ["AWS_REGION"] = region
llm_response = ""

aws = boto3.session.Session(profile_name="genaiday", region_name=region)
client = aws.client("bedrock-runtime")
