## Prerequisites

* An AWS account
* [Docker](https://docs.docker.com/install/)
* [AWS Command Line Interface](https://docs.aws.amazon.com/cli/latest/userguide/install-cliv1.html)


## Setting up AWS user permissions

- AmazonEC2ContainerRegistryFullAccess
- AmazonS3FullAccess
- AWSBatchFullAccess
- create the following policy:

{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "ecr-public:GetAuthorizationToken",
                "sts:GetServiceBearerToken",
                "ecr-public:BatchCheckLayerAvailability",
                "ecr-public:GetRepositoryPolicy",
                "ecr-public:DescribeRepositories",
                "ecr-public:DescribeRegistries",
                "ecr-public:DescribeImages",
                "ecr-public:DescribeImageTags",
                "ecr-public:GetRepositoryCatalogData",
                "ecr-public:GetRegistryCatalogData",
                "ecr-public:UploadLayerPart",
                "ecr-public:InitiateLayerUpload",
                "ecr-public:CompleteLayerUpload",
                "ecr-public:PutImage"
            ],
            "Resource": "*"
        }
    ]
}





## Setting up elastic container service user role

The example scripts used in this tutorial will download and upload files from an S3 bucket. To allow for this we need to create an appropriate role to be used by the batch job.

1. Navigate to the IAM service.
2. Click on "roles" and then "Create Role"
3. Under "Select type of trusted entity" select "AWS service"
4. Under "Choose a use case" select "Elastic Container Service"
5. Under "Select your user case" select "Elastic Container Service Task, Allows ECS tasks to call AWS services on your behalf."
6. Click "Next: Permissions"
7. Type "S3" into the filter policies box
8. Check the box next to "AmazonS3FullAccess"
9. Click "Next: Tags" and then "Next: Review"
10. Enter a role name such as "ecs_role" and click "Create role"

## Configuring AWS CLI

If you have not already done so you will need to configure AWS CLI to your account.

1. Install AWS CLI, there is a link in the Prerequisites section.
2. Navigate to the users section of the IAM dashboard.
3. Select the user you configured in the previous section.
4. Click the "Security credentials" tab.
5. Click "Create access key". Copy down both the Access key ID and secret access key to a secure location.
6. Open a terminal on your local machine and enter `aws configure`
7. Enter the access key and secret access key from Step 4. Choose a default region name and output format, the defaults are fine if you are not sure.

## Preparing your Python scripts

Each time you submit a batch job you will be able to pass a different set of parameters to your Python script. In this project we send those parameters using the strategy_optimizer notebook, there is an explanation of each parameter.

## Create an S3 bucket to store results

An S3 bucket is used to store files that will either be downloaded and used by your python script, and to store the results that the python script produces.

1. Navigate to the S3 service in AWS console.
2. Click "Create Bucket".
3. Entire a bucket name (e.g., backtesterbucket). Note that this bucket name must be unique across all existing S3 buckets.
4. All remaining settings can remain at their default value, click through and create bucket.
5. Select your newly created bucket in the "Buckets" section
6. Create appropriately named folders to contain results.



## Building your Docker image and uploading to ECR

Each batch job will be pointed toward the docker image it will run. This tutorial will push our docker image to the AWS Elastic Container Repository.

A Dockerfile is provided for this project, this will have to be adapted or replaced when running your own code.

Docker images can very quickly grow in size, the one used here will be about 170MB. If your internet is slow this can make the pushes to ECR painful. A second option is to upload or clone your project onto an EC2 instance and build/push your image there. If you are dealing with slow upload speeds this will make the process significantly quicker.

1. Navigate to the ECR service in the AWS console.
2. Select Create repository.
3. Give your repository a name, this project will use aws-backtester-repository.
4. After creating the repository, you will see a SEE PUSH COMMANDS in the repository (top right of the screen). Follow those steps
6. You will need a token for these, so remember to create all the permissions required.

## Configuring AWS Batch

### Compute Environment

The compute environment describes the EC2 instances that will be provisioned when running batch jobs.

1. Select the "Compute Environments" section and select "Create environment"
2. Leave the compute type as "Managed".
3. Specify any name for this environmental, for example "batch_backtester_m5a".
4. Select "Create new role" for service and instance roles. If you have previously created roles from this step, or other appropriately configured roles, you can select those.
5. Select or create an EC2 key pair, you will not need this key pair for the remainder of this tutorial.
6. Remove "optimal" from allowed instance types and select "m5a.large". Note that this instance will cost $0.086/hr to run, but the test scrits used here should only run for seconds.
7. **Make sure "Minimum vCPUs" is set to 0. If it is not AWS will keep instances running, and charging you, even if no jobs are running.**
8. Click create, the rest of the options can remain at their default values.

### Queue

The queue is responsible for accepting new job submissions and scheduling them onto the compute environment.

1. Select the "Job Queues" section and click "Create job queue".
2. Specify a name for the queue, (e.g., "batch_backtester_queue").
3. Specify any number for priority (e.g., 1).
4. Select the compute environment you created in the last section and click "Create job queue"

### Job Definition Template

A job definition specifies the parameters for a particular job. The parameters defined include a link to the docker image that will be run, the commands to run, memory/cpu/gpu requirements, and any environmental variables.

Here we will create a job definition in the AWS console that will act as a template for future job definitions. Job submissions will point to this particular job definition template, and then define any modifications to the parameters that particular task.

1. Select "Job Definitions" and click "create"
2. Enter any job name (e.g., "batch_backtester_job_definition")
3. Enter 60 format the execution timeout, this will automatically kill the job if it runs for more than 60 seconds. You will likely have to enter a much higher value when using your own code
4. For Job Role, select the select container service user role created earlier (e.g., ecs_role)
5. For container image, paste in the ECR URI that you generated in the "Building your Docker Image..." section, for example `<Account ID>.dkr.ecr.us-east-1.amazonaws.com/aws-backtester-repository`
6. For command, enter `./run_job.sh`. This command will be executed inside the docker container once it begins.
7. Enter 1 for vCPUs and 1000 for Memory(MiB).
8. Leave the remaining values at their default and select Create Job Definition

## Defining Jobs

JSON files are used to define job submissions. Each job JSON file used here will define a job name, point to the queue and job definition we just created, and define any desired arguments to pass to the Python script. This will be done in strategy_optimizer

AWS Batch allows you to define environment variables for each job submission. The `run_job.sh` script which will be called at the start of our job will convert those environment variables to command line arguments for the python script. This is done by parsing the `python script.py --help` command, whose output is created by the `argparse` package. To pass command line arguments to the script you must specify environment variables in the JSON file with names that match the desired command line argument name.

The contents of `sample_job_1.sh` are shown below. You need to modify several values:

1. `jobName` can be any string you desire.
2. `jobQueue` needs to be the name of the queue you created previously
3. `jobDefinition` is the name of the job definition template created earlier.
4. Finally, set you can the desired environment variables that will be used by the bash script and/or passed as command line arguments to the Pythons script. In our sample script `save_name` is the name assigned to the results and log files. Here we also assign a new value to `var_a`. All other parameters in the script will take on their default values defined in `parse_arg()`.




## Monitoring jobs

In the AWS console you can navigate to Batch service and select "Dashboard". This will allow you to monitor the status of each of your jobs, note that it may take several minutes before Batch provisions the compute instances and executes the job.

By clicking the numbers under each type of job status (e.g., "Succeeded") you can see a list of all jobs with this status. You can see a job's details by clicking the Job ID, if you scroll to the bottom of these details there will be a link view the Cloudwatch Logs.

## Retrieving results

Once a job is complete you can retrieve the results one of two ways. First, you can navigate to the S3 bucket in the AWS console and view or download the files. Alternatively, you can use the download function to gather all the job results and concatenate them in one single dataframe. This is done in the strategy_optimizer notebook



## Cleaning up

You will be charged for the memory your docker images use when uploaded to the ECR, and for the files contained in your S3 bucket. You will be continuosly charged for these resources even when not running jobs. To avoid this make sure you delete your unused ECR repositories and S3 buckets/files once finished.
