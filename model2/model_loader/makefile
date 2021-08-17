DOCKER_REPOSITORY := kooose/ml-system-in-actions

ABSOLUTE_PATH := $(shell pwd)

DOCKERFILE := Dockerfile
IMAGE_VERSION := 0.0.1

PARAMETER_BASED_PATTERN := model_loader_iris_tree

.PHONY: build
build:
	docker build \
		-t $(DOCKER_REPOSITORY):$(PARAMETER_BASED_PATTERN)_$(IMAGE_VERSION) \
		-f ./$(DOCKERFILE) \
		.
	
.PHONY: push
push:
	docker push $(DOCKER_REPOSITORY):$(PARAMETER_BASED_PATTERN)_$(IMAGE_VERSION)