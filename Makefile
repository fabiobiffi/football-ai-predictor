export DOCKER_DEFAULT_PLATFORM=linux/amd64
COMPOSE_DEV=docker-compose.yml

# DOCKER TASKS
dev-up: ## Spin up compose
	docker compose -f ${COMPOSE_DEV} up

dev-up-build: dev-down  ## Build and spin up compose
	docker compose -f ${COMPOSE_DEV} down -v
	docker compose -f ${COMPOSE_DEV} up --build

dev-down: ## Spin down compose
	docker compose -f ${COMPOSE_DEV} down -v
