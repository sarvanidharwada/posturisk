#!/bin/sh
export GIT_AUTHOR_NAME="Sarvani Dharwada"
export GIT_AUTHOR_EMAIL="sarvanidharwada@users.noreply.github.com"
export GIT_COMMITTER_NAME="Sarvani Dharwada"
export GIT_COMMITTER_EMAIL="sarvanidharwada@users.noreply.github.com"
# Tell git to rewrite all commits on all branches with these env vars
git filter-branch -f --env-filter '
    export GIT_AUTHOR_NAME="Sarvani Dharwada"
    export GIT_AUTHOR_EMAIL="sarvanidharwada@users.noreply.github.com"
    export GIT_COMMITTER_NAME="Sarvani Dharwada"
    export GIT_COMMITTER_EMAIL="sarvanidharwada@users.noreply.github.com"
' -- --all
