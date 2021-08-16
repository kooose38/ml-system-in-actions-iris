DOCKER_REPOSITORY := kooose/ml-system-in-actions

ABSOLUTE_PATH := $(shell pwd)

DOCKERFILE := Dockerfile
IMAGE_VERSION := 0.0.1

.PHONY: build_client
build_client:
	docker build \
		-t $(DOCKER_REPOSITORY):client_$(IMAGE_VERSION) \
		-f $(DOCKERFILE).client \
		.

.PHONY: push_client
push_client:
	docker push $(DOCKER_REPOSITORY):client_$(IMAGE_VERSION)

.PHONY: deploy
deploy:
	istioctl install -y
	kubectl apply -f manifests/namespace.yml
	kubectl apply -f manifests/