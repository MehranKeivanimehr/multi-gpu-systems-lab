# Security and shared-host execution

Report a suspected problem using the repository's private security reporting facility when available, or contact Mehran Keivanimehr directly. No contact address is assigned in this repository.

The reference campaigns contain host-specific device selections. Review them on the target machine before starting a GPU workload. Use the supplied launchers and preflight gates, and obtain authorization for the selected devices from the cluster owner. Historical A100 notebook cells are retained for provenance; the cleaned notebook is the reproduction entry point.

Offline validation and report generation do not access GPUs or the network. CI executes only that offline workflow.
