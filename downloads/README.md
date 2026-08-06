# Web download catalog

The browser application's **Downloads** page is generated from JSON records in
`downloads/metadata/`. Binary installers and publication PDFs may remain outside
Git; a card is enabled whenever its `repo_path` exists on the deployed server.

Each metadata file uses schema `microseg.download.v1` and provides:

- `asset_id`: stable URL identifier;
- `display_name`, `help_text`, and optional `description`;
- `category`, optional `version`, `featured`, and `order`;
- `repo_path`: repository-relative file path (paths outside the repository are rejected);
- optional `download_name` and `media_type`.

The page computes the file size and SHA-256 checksum at request time. Missing
cataloged files remain visible as unavailable so deployment omissions are clear.
