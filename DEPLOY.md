# Railway Deployment Instructions

## Step 1: Create GitHub Repository

1. Go to GitHub.com and create a new repository
2. Repository name: `policy-radar-railway` (or your preferred name)
3. Make it Public or Private (your choice)
4. **Don't** initialize with README (we already have one)

## Step 2: Push to GitHub

After creating the repository, run these commands in the `railway-app` directory:

```bash
# Add GitHub remote (replace USERNAME and REPO_NAME with your values)
git remote add origin https://github.com/USERNAME/REPO_NAME.git

# Push to GitHub
git branch -M main
git push -u origin main
```

## Step 3: Deploy to Railway

1. Go to Railway.app
2. Create new project
3. Select "Deploy from GitHub repo"
4. Choose your `policy-radar-railway` repository
5. Railway should automatically detect the Python app and deploy

## Expected Railway Detection

Railway should automatically:
- ✅ Detect Python from `requirements.txt`
- ✅ Use Python 3.11 from `runtime.txt`
- ✅ Install dependencies with `pip install -r requirements.txt`
- ✅ Start with command from `Procfile`: `uvicorn main:app --host 0.0.0.0 --port $PORT`

## Health Check Endpoints

Once deployed, test these endpoints:
- `https://your-app.railway.app/` - API info
- `https://your-app.railway.app/health` - Health check
- `https://your-app.railway.app/api/health` - API health check