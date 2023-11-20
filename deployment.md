# Install infrastructure

## Setting Bedrock permissions

Currently (October 2023), Bedrock usage regions are as follows.

![image](https://github.com/kyopark2014/question-answering-chatbot-with-vector-store/assets/52392004/1690aaab-5e1e-4c27-b4a2-1fd3cabf536c)

Here we use us-east-1 (N. Virginia). Access [Model access](https://us-east-1.console.aws.amazon.com/bedrock/home?region=us-east-1#/modelaccess) and select [Edit] to edit all models. Make it available. In particular, Anthropic Claude and "Titan Embeddings G1 - Text" must be available for LLM and Vector Embedding.

![image](https://github.com/kyopark2014/question-answering-chatbot-with-vector-store/assets/52392004/112fa4f6-680b-4cbf-8018-3bef6514ccf3)



## Installing infrastructure using CDK


Here, the infrastructure is installed using [AWS CDK](https://aws.amazon.com/ko/cdk/) in [Cloud9](https://aws.amazon.com/ko/cloud9/).

1) Access [Cloud9 Console](https://ap-northeast-2.console.aws.amazon.com/cloud9control/home?region=ap-northeast-2#/create) and click [Create environment]-[Name ], enter the name “chatbot” and select “m5.large” for the EC2 instance. Leave the rest as default, scroll to the bottom and choose Create.

![noname](https://github.com/kyopark2014/chatbot-based-on-Falcon-FM/assets/52392004/7c20d80c-52fc-4d18-b673-bd85e2660850)

2) After [Open] “chatbot” in [Environment](https://ap-northeast-2.console.aws.amazon.com/cloud9control/home?region=ap-northeast-2#/), do the following: Run terminal.

![noname](https://github.com/kyopark2014/chatbot-based-on-Falcon-FM/assets/52392004/b7d0c3c0-3e94-4126-b28d-d269d2635239)

3) Change EBS size

Download the script as shown below.

```text
curl https://raw.githubusercontent.com/kyopark2014/technical-summary/main/resize.sh -o resize.sh
```

Afterwards, change the capacity to 80G using the command below.
```text
chmod a+rx resize.sh && ./resize.sh 80
```


4) Download the source.

```java
git clone https://github.com/kyopark2014/korean-chatbot-using-amazon-bedrock
```

5) Go to the cdk folder and install the required libraries.

```java
cd korean-chatbot-using-amazon-bedrock/cdk-korean-chatbot/ && npm install
```

6) Perform Boostraping to use CDK.

Check your Account ID with the command below.

```java
aws sts get-caller-identity --query Account --output text
```

Perform bootstrap as shown below. Here, “account-id” is the 12-digit Account ID confirmed with the above command. You only need to run bootstrap once, so if you were already using cdk, you can skip bootstrap.

```java
cdk bootstrap aws://account-id/ap-northeast-2
```

8) Install the infrastructure.

```java
cdk deploy --all
```
9) Once installation is complete, check the WebUrl in the browser as shown below and connect using the browser.