import express from "express";
import { getConnection } from "./db.js";

const app = express();

app.use(express.json());

/*
    MCP Tool:
    get_products(category)
*/
app.get("/products/:categoryId", async (req, res) => {

    try {

        const categoryId = req.params.categoryId;

        const db = await getConnection();

        const result = await db.request().input("categoryId", categoryId)
            .query(`
                SELECT *    
                FROM Products
                WHERE CategoryId = @categoryId
                ORDER BY Price DESC
            `);

        res.json({
            tool: "get_products",
            categoryId: categoryId,
            count: result.recordset.length,
            products: result.recordset
        });

    } catch (error) {

        console.error(error);

        res.status(500).json({
            error: error.message
        });
    }
});

app.listen(3000, () => {

    console.log("MCP Server running on port 3000");
});