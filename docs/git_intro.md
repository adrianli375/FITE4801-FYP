# Motivation to use Git and GitHub

Git is a powerful version control system used for managing software development projects. It allows multiple developers to collaborate on a project while keeping track of changes to the codebase over time. However, if you are new to Git, it can be overwhelming and confusing to get started. This introduction will provide you with a basic understanding of what Git is and how to use it to manage your project's codebase.

At its core, Git is a tool that allows you to save different versions of your code over time, create branches for experimenting with new features, and merge changes made by different developers. With Git, you can collaborate with others without worrying about losing your work or overwriting someone else's changes. It also enables you to revert to a previous version of your code if something goes wrong.

To get started with Git, you'll need to install it on your computer and set up a repository for your project. A repository is a directory that contains your project's files and all the information Git needs to manage the changes to those files. You can create a repository from scratch, or you can clone an existing one from a remote location like GitHub or Bitbucket.

Once you have a repository set up, you can use Git commands to add, commit, and push changes to your code. These commands allow you to keep track of the changes you've made, write clear commit messages, and share your changes with others. With Git, you can also create branches to work on new features or experiment with different ideas without affecting the main codebase.

While Git can be intimidating at first, it's an essential tool for anyone working on software development projects. By understanding the basics of Git, you'll be able to manage your codebase effectively and collaborate with others to create high-quality software.

# How to install git through command line ([Reference](https://github.com/UBC-CS/cpsc330-2022W2/blob/main/docs/git_installation.md))

## Setting up

GitHub is a web-based application and does not require set-up. Follow the instructions below for setting up git. 

#### Mac Users

Open Terminal (Applications –> Utilities folder or search with Spotlight). From the terminal, run the command:

```
xcode-select --install
```

This will install git and many other very useful applications as well (including Make).

#### Ubuntu Users

Open the terminal and install git using your system package manager. For example

```
sudo apt-get install git
```

should do the trick on Ubuntu.

#### Windows Users

Go to http://git-scm.com. Click on the download link, and accept all defaults in the installation process. 
Installing git will also install for you a minimal UNIX environment with a bash shell and terminal window. 

If the above does not work for you, follow the [installation instructions here](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git).

## Testing your git installation

Open your Terminal (or Git for Windows) and run `git --version`.  
If you are returned the version of git, it means your install was successful!

# How to begin using this Git repo

Choose a folder/location in your local device. Once you selected your folder location, open the terminal in your device. Run `git clone <repo-link>` in your local device. For example, to get a local repository for this Git repo, you can run the command `git clone https://github.com/adrianli375/Test-Project`. 

Since you will become a (potential) contributor of this repo, you would like to work separately with my work and to avoid conflicts with me while editing your files. To achieve this, you can create your own branch by the command `git branch <branch-name>`. For example, if you would like to set your branch name to `branch-1`, you can run the command `git branch branch-1`. To switch to the branch that you are working with, run `git checkout <branch-name>`. For example, if your branch name is `branch-1`, you can run the command `git checkout branch-1`. 

Please ensure that you are working on a branch that is different from `main`. If you are working with the `main` branch, you risk yourself the chance of your files being overwritten by the files in the `main` branch. 

After changing the branch to a suitable one (NOT the main branch), you can pull the files from this repo by running the command `git pull origin main` at your chosen folder location. 

If you finished editing in your own local repo, you can run the following commands to update your changes in your local branch: 
1. `git init`: Initialises the `.gitignore` file in your local repository. Run this **only for the first time** before your commit. 
2. `git add .`: Adds the file you would like to stage in your commit. 
3. `git commit -m "<message>"`: Commits your changes with a pre-specified message. For example, if the message is "changes updated in a file", the command to be typed is `git commit -m "changes updated in a file"`. 
4. `git push origin <branch-name>`: Push your changes to the branch you specified. For example, if your branch name is `branch-1`, the command to be typed is `git push origin branch-1`. 

Finally, if you are the main contributors of the git repo, you can merge branches to the main branch to update to the latest changes. You can achieve this by the following steps: 
1. `git checkout main`: Switch the branch to the `main` branch
2. `git merge <branch-name>`: Merge changes from the working branch to the `main` branch. For example, if the branch that you are currently working on is `branch-1`, you should run the command `git merge branch-1`. 

# Common Git Commands

Here are some basic Git commands that a beginner should use:

`git init`: Initializes a new Git repository in your current directory. This creates a hidden .git directory that contains all of the version control information for your project.

`git add`: Adds a file to the staging area, which is a place to store changes you want to commit. Use `git add <filename>` to add a specific file or `git add .` to add all files in the current directory.

`git commit`: Records the changes you made to the staging area and creates a new commit in the Git history. Use `git commit -m "commit message"` to create a new commit with a message describing the changes.

`git status`: Shows the current status of your repository, including which files have been modified, which files are staged for commit, and which branch you're on.

`git log`: Shows a log of all the commits in your repository, including the commit message, author, and date.

`git branch`: Lists all the branches in your repository. Use git branch <branch-name> to create a new branch, and git checkout <branch-name> to switch to a different branch.

`git pull`: Updates your local repository with the latest changes from a remote repository. When working on a project with multiple developers, it's important to keep your local branch up to date with the changes others have made to the codebase. With git pull, you can fetch changes from the remote repository and merge them into your local branch. This ensures that you're working with the most up-to-date code and minimizes the risk of conflicts when working with others. The syntax for git pull is `git pull <remote> <branch>`, where <remote> is the name of the remote repository and `<branch>` is the branch you want to merge into your local branch.

`git push`: Send changes from your local repository to a remote repository. When you've made changes to the code and want to share those changes with others or back up your work, you can use git push to update the remote repository with your local changes. This ensures that everyone working on the project has access to the latest code and reduces the risk of losing your work. The syntax for git push is `git push <remote> <branch>`, where <remote> is the name of the remote repository and <branch> is the branch you want to push your changes to.
