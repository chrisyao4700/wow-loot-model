FROM public.ecr.aws/lambda/python:3.12

COPY pyproject.toml README.md ${LAMBDA_TASK_ROOT}/
COPY src ${LAMBDA_TASK_ROOT}/src
COPY artifacts ${LAMBDA_TASK_ROOT}/artifacts

RUN pip install --no-cache-dir ${LAMBDA_TASK_ROOT}

CMD ["sea_of_blossoms_model.lambda_handler.handler"]
