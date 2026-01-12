# ===================================================================
# VectorMesh - Remote Transfer Makefile
# ===================================================================
# This Makefile provides commands for transferring assets and
# artefacts between local and remote servers using SCP.
# ===================================================================

# Configuration Variables
# Customize these for your environment
USERNAME := rgrouls
IP := 145.38.191.144
REMOTE_FOLDER := /home/rgrouls/vectormesh

# Local directories
LOCAL_ASSETS := assets
LOCAL_ARTEFACTS := artefacts

# Remote directories (relative to REMOTE_FOLDER)
REMOTE_ASSETS := $(REMOTE_FOLDER)/assets
REMOTE_ARTEFACTS := $(REMOTE_FOLDER)/artefacts

# SCP options
SCP_OPTS := -r -C -p

# ===================================================================
# Targets
# ===================================================================

.PHONY: help scp-push scp-pull check-config test-connection

# Default target - show help
help:
	@echo "====================================================================="
	@echo "VectorMesh Remote Transfer Commands"
	@echo "====================================================================="
	@echo ""
	@echo "Configuration:"
	@echo "  USERNAME       : $(USERNAME)"
	@echo "  IP             : $(IP)"
	@echo "  REMOTE_FOLDER  : $(REMOTE_FOLDER)"
	@echo ""
	@echo "Available targets:"
	@echo "  make help              - Show this help message"
	@echo "  make check-config      - Verify configuration variables"
	@echo "  make test-connection   - Test SSH connection to remote server"
	@echo ""
	@echo "Push (Local → Remote):"
	@echo "  make scp-push                - Push all aktes_* folders from local assets/"
	@echo "  make scp-push FOLDER=name    - Push assets/name to remote assets/"
	@echo ""
	@echo "Pull (Remote → Local):"
	@echo "  make scp-pull                - Pull all files from remote artefacts/"
	@echo "  make scp-pull FOLDER=name    - Pull remote artefacts/name to local artefacts/"
	@echo ""
	@echo "Examples:"
	@echo "  make scp-push                                    # Push all aktes_* folders"
	@echo "  make scp-push FOLDER=aktes_theshold_50_d97342   # Push assets/aktes_theshold_50_d97342"
	@echo "  make scp-pull                                    # Pull all remote artefacts"
	@echo "  make scp-pull FOLDER=results_2024                # Pull remote artefacts/results_2024"
	@echo ""
	@echo "====================================================================="

# Check configuration
check-config:
	@echo "Checking configuration..."
	@if [ "$(IP)" = "your.server.ip" ]; then \
		echo "ERROR: Please set IP variable"; \
		echo "Usage: make scp-push IP=your.server.ip"; \
		exit 1; \
	fi
	@if [ -z "$(USERNAME)" ]; then \
		echo "ERROR: USERNAME not set"; \
		exit 1; \
	fi
	@if [ -z "$(REMOTE_FOLDER)" ]; then \
		echo "ERROR: REMOTE_FOLDER not set"; \
		exit 1; \
	fi
	@echo "Configuration OK"
	@echo "  Target: $(USERNAME)@$(IP):$(REMOTE_FOLDER)"

# Test SSH connection
test-connection: check-config
	@echo "Testing connection to $(USERNAME)@$(IP)..."
	@ssh -o ConnectTimeout=5 -o BatchMode=yes $(USERNAME)@$(IP) "echo 'Connection successful!'" || \
		(echo "ERROR: Cannot connect to server. Check IP, USERNAME, and SSH keys."; exit 1)

# Push folders to remote assets directory
# Usage: make scp-push [FOLDER=folder_name]
# If FOLDER is set, push only that folder. Otherwise push all aktes_* folders.
scp-push: check-config
	@if [ ! -d "$(LOCAL_ASSETS)" ]; then \
		echo "ERROR: Local assets directory not found: $(LOCAL_ASSETS)"; \
		exit 1; \
	fi
	@# Create remote assets directory if it doesn't exist
	@echo "Ensuring remote directory exists..."
	@ssh $(USERNAME)@$(IP) "mkdir -p $(REMOTE_ASSETS)"
ifdef FOLDER
	@echo "====================================================================="
	@echo "Pushing $(LOCAL_ASSETS)/$(FOLDER) to $(USERNAME)@$(IP):$(REMOTE_ASSETS)/"
	@echo "====================================================================="
	@if [ ! -d "$(LOCAL_ASSETS)/$(FOLDER)" ]; then \
		echo "ERROR: Folder not found: $(LOCAL_ASSETS)/$(FOLDER)"; \
		exit 1; \
	fi
	@echo "Transferring $(FOLDER)..."
	@scp $(SCP_OPTS) "$(LOCAL_ASSETS)/$(FOLDER)" $(USERNAME)@$(IP):$(REMOTE_ASSETS)/
	@echo ""
	@echo "Transfer complete!"
else
	@echo "====================================================================="
	@echo "Pushing all aktes_* folders to $(USERNAME)@$(IP):$(REMOTE_ASSETS)/"
	@echo "====================================================================="
	@FOLDERS=$$(find $(LOCAL_ASSETS) -maxdepth 1 -type d -name "aktes_*" 2>/dev/null); \
	if [ -z "$$FOLDERS" ]; then \
		echo "WARNING: No folders matching 'aktes_*' found in $(LOCAL_ASSETS)"; \
	else \
		for folder in $$FOLDERS; do \
			echo "Transferring $$folder..."; \
			scp $(SCP_OPTS) "$$folder" $(USERNAME)@$(IP):$(REMOTE_ASSETS)/ || exit 1; \
		done; \
		echo ""; \
		echo "Transfer complete!"; \
	fi
endif

# Pull artefacts from remote to local
# Usage: make scp-pull [FOLDER=folder_name]
# If FOLDER is set, pull only that folder. Otherwise pull all files.
scp-pull: check-config
	@# Create local artefacts directory if it doesn't exist
	@mkdir -p $(LOCAL_ARTEFACTS)
	@# Check if remote directory exists
	@echo "Checking remote directory..."
	@ssh $(USERNAME)@$(IP) "[ -d $(REMOTE_ARTEFACTS) ]" || \
		(echo "ERROR: Remote artefacts directory not found: $(REMOTE_ARTEFACTS)"; exit 1)
ifdef FOLDER
	@echo "====================================================================="
	@echo "Pulling $(FOLDER) from $(USERNAME)@$(IP):$(REMOTE_ARTEFACTS)/"
	@echo "====================================================================="
	@echo "Transferring $(FOLDER)..."
	@scp $(SCP_OPTS) $(USERNAME)@$(IP):$(REMOTE_ARTEFACTS)/$(FOLDER) $(LOCAL_ARTEFACTS)/ || \
		(echo "ERROR: Folder not found or transfer failed"; exit 1)
	@echo ""
	@echo "Transfer complete! Files saved to $(LOCAL_ARTEFACTS)/"
else
	@echo "====================================================================="
	@echo "Pulling all files from $(USERNAME)@$(IP):$(REMOTE_ARTEFACTS)/"
	@echo "====================================================================="
	@echo "Transferring artefacts/*..."
	@scp $(SCP_OPTS) $(USERNAME)@$(IP):$(REMOTE_ARTEFACTS)/* $(LOCAL_ARTEFACTS)/ || \
		(echo "WARNING: No files found or transfer failed"; exit 0)
	@echo ""
	@echo "Transfer complete! Files saved to $(LOCAL_ARTEFACTS)/"
endif

# Utility target to list remote files
list-remote: check-config
	@echo "====================================================================="
	@echo "Remote Directory Listing"
	@echo "====================================================================="
	@echo ""
	@echo "Assets ($(REMOTE_ASSETS)):"
	@ssh $(USERNAME)@$(IP) "ls -lh $(REMOTE_ASSETS) 2>/dev/null || echo 'Directory not found'"
	@echo ""
	@echo "Artefacts ($(REMOTE_ARTEFACTS)):"
	@ssh $(USERNAME)@$(IP) "ls -lh $(REMOTE_ARTEFACTS) 2>/dev/null || echo 'Directory not found'"

# Utility target to list local files
list-local:
	@echo "====================================================================="
	@echo "Local Directory Listing"
	@echo "====================================================================="
	@echo ""
	@echo "Assets ($(LOCAL_ASSETS)):"
	@ls -lh $(LOCAL_ASSETS) 2>/dev/null || echo "Directory not found"
	@echo ""
	@echo "Artefacts ($(LOCAL_ARTEFACTS)):"
	@ls -lh $(LOCAL_ARTEFACTS) 2>/dev/null || echo "Directory not found"
