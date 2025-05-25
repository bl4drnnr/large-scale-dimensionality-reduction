<h1 align="center">Large Scale Dimensionality Reduction</h1>

# Table of Contents
1. [Introduction](#introduction)
2. [Project setup](#project-setup)

# Introduction

This project is an extension of the already existing project named [Text Embedding Visualization Dashboard](https://github.com/bl4drnnr/text-embedding-visualization-dashboard). Project, we intend to create embeddings from one of the given data sets, save them in a vector database and then visualize them using an application created in the streamlit library, using the given visualization methods. Wrap the solution in a Docker image, which will make it easier to run the application regardless of the environment.

But there are 2 main problem that have been discovered during the implementation phase:

- The first problem was that the size of this container was 20-25GB when the entire application is executed on the local computer.
- The second and the most important problem has been discovered during the process of implementation of one of the application’s features, which is also the cause of the first issue. The application allows users to upload their custom datasets. It means that the apart from the datasets, which can be large, users will have to store them in the vector database and use resources of their machine to perform the dimensionality reduction.

Therefore, in order to solve two issues that have been mentioned above, a kind of module needs to be implemented and deployed, that is going to take some of the responsibility from the end-user device to a remote server. Due to the fact that dimensionality reduction suffers from high computational complexity, the solution here is going to be any high-performance computing (HPC) environments.

This approach can help to solve both of the issues. As it was mentioned, the problem with the size of the container lies mostly not in the Docker containerization technology itself, but in the fact that datasets and vector databases require lots of free space. The HPC environments can propose much more space than end-user machines, which will save the issue with the storage and free space.

By the definition of the HPC as high-performance computing environments, it has special hardware/software combined with the huge amount of resources such as RAM, CPUs and GPUs that are used together to perfrom very complex calculations, where dimensionality reduction is a one of the examples.

To sum it up: super computers are going to solve two main issues with the already existing application - it’s going to take a responsibility for storing data along with the computational costs of differente dimensionality techniques.

# Project setup

Due to the fact that this project is using **uv** as a package manager, there is a little different sequence of commands that you need to use in order to set it up locally:

```sh
uv init large-scale-dimensionality-reduction --python 3.12
mv large-scale-dimensionality-reduction/* .
rm -rf large-scale-dimensionality-reduction
rm main.py
uv venv .venv
source .venv/bin/activate
uv sync
```

Once the project is set up, you need to create the `.env` file in the root of the project and put the following information:

```txt
CHROMA_HOST=<IP address>
CHROMA_PORT=8000

AWS_SECRET_ACCESS_KEY=<your aws secret key>
AWS_ACCESS_KEY_ID=<your aws secret id>
AWS_REGION=<aws region>
S3_BUCKET_NAME=<bucket name>
```

The very last step that needs to be done before you can start the project is going to be the setup of the ChromaDB in the cloud (AWS in our case). In order to do that you need to run the following commands in your terminal using AWS CLI:

```sh
# This is going to set up the cloudformation in the AWS. Instead of template URL you can see chroma_aws_cloudformation.json in the root folder of the project.
aws cloudformation create-stack --stack-name my-chroma-stack --template-url https://s3.amazonaws.com/public.trychroma.com/cloudformation/latest/chroma.cf.json

# Using this command you will be able to obtain an IP address that you can place in the .env file
aws cloudformation describe-stacks --stack-name my-chroma-stack --query 'Stacks[0].Outputs'
```

The CloudFormation template allows you to pass particular key/value pairs to override aspects of the stack. Available keys are:

- `InstanceType` - the AWS instance type to run (default: `t3.small`)
- `KeyName` - the AWS EC2 KeyPair to use, allowing to access the instance via SSH (default: `none`)

To set a CloudFormation stack's parameters using the AWS CLI, use the `--parameters` command line option. Parameters must be specified using the format `ParameterName={parameter},ParameterValue={value}`.

To get more information on how you can set up a ChromaDB in the cloud, please, check out [ChromaDB AWS Deployment](https://docs.trychroma.com/production/cloud-providers/aws) in the [References](#references) section.

And then, simple start the project (from the root folder or you can start [`frontend.py`](src/large_scale_dimensionality_reduction/frontend/frontend.py) directly):

```sh
streamlit run src/large_scale_dimensionality_reduction/frontend/frontend.py
```

# References

- [ChromaDB AWS Deployment](https://docs.trychroma.com/production/cloud-providers/aws)